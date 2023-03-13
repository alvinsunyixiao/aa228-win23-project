import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import typing as T

from jaxopt import OSQP, BoxOSQP
from jax import lax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from dynamics import AffineCtrlSys


class MLPCertificate(nn.Module):
    configs: T.Tuple[int, ...]
    prior: T.Callable = lambda x: 0

    @nn.compact
    def __call__(self, x):
        prior = self.prior(x)

        for i, feat in enumerate(self.configs):
            x = nn.Dense(features=feat)(x)
            if i < len(self.configs) - 1:
                x = nn.tanh(x)

        return 0.5 * jnp.sum(x * x, axis=-1) + prior


class CLBF:
    def __init__(self,
        dynamics: AffineCtrlSys,
        clf_lambda: float = 1.,
        dt: float = 1e-2,
        safe_level: float = 1.,
        mlp_configs: T.Tuple[int, ...] = (64, 64, 64),
    ) -> None:
        self.dynamics = dynamics
        self.clf_lambda = clf_lambda
        self.dt = dt
        self.safe_level = safe_level

        # initialize mlp
        self.mlp_cert = MLPCertificate(configs=mlp_configs)
        rng = jax.random.PRNGKey(0)
        params = self.mlp_cert.init(rng, jnp.ones((1, self.dynamics.state_dim)))["params"]
        tx = optax.sgd(optax.exponential_decay(1e-3, 2000, 0.1), 0.9)
        self.state = TrainState.create(apply_fn=self.mlp_cert.apply, params=params, tx=tx)

        # checkpoint manager
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def save(self, ckpt_dir: str, step: int = 0, keep: int = 10):
        save_checkpoint(ckpt_dir=ckpt_dir,
                        target=self.state,
                        step=step,
                        keep=keep,
                        overwrite=True,
                        orbax_checkpointer=self.checkpointer)

    def load(self, ckpt_dir: str, step: T.Optional[int] = None):
        self.state = restore_checkpoint(ckpt_dir=ckpt_dir,
                                        target=self.state,
                                        step=step,
                                        orbax_checkpointer=self.checkpointer)

    def V(self, state):
        return self.state.apply_fn({"params": self.state.params}, state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(self, state, u_ref=None, relaxation_penalty=1e5):
        sol, (V, Vdot) = self.solve_CLBF_QP(self.V, state, u_ref=u_ref, relaxation_penalty=relaxation_penalty)
        return sol.params.primal[:-1], (V, Vdot, sol.params.primal[-1])

    def solve_CLBF_QP(self, V_func, state, relaxation_penalty=1e4, u_ref=None):
        V, V_D_x_s = jax.value_and_grad(V_func)(state)

        Lf_V = V_D_x_s @ self.dynamics.f(state)
        Lg_V_c = V_D_x_s @ self.dynamics.g(state)

        ur_dim = self.dynamics.control_dim + 1

        # objective
        Q = 2. * jnp.eye(ur_dim)
        Q = Q.at[-1, -1].set(0.)
        if u_ref is None:
            u_ref = jnp.zeros(self.dynamics.control_dim)
        c = jnp.concatenate([-2. * u_ref, jnp.array([relaxation_penalty])])

        # inequality
        G1 = jnp.concatenate([Lg_V_c, jnp.array([-1.])])
        h1 = -self.clf_lambda*V - Lf_V
        G2 = jnp.zeros(ur_dim)
        G2 = G2.at[-1].set(-1.)
        h2 = 0.
        G = jnp.stack([G1, G2])
        h = jnp.stack([h1, h2])
        # control limits
        G3 = jnp.zeros((self.dynamics.control_dim, ur_dim))
        G3 = G3.at[:, :-1].set(jnp.eye(self.dynamics.control_dim))
        ulim_low, ulim_high = self.dynamics.control_limit()
        G = jnp.concatenate([G, -G3, G3], axis=0)
        h = jnp.concatenate([h, -ulim_low, ulim_high], axis=0)

        # solve OSQP
        qp = OSQP(eq_qp_solve="cg+jacobi")
        sol = qp.run(params_obj=(Q, c), params_ineq=(G, h))

        u = sol.params.primal[:-1]
        Vdot = Lf_V + Lg_V_c @ u

        return sol, (V, Vdot)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _clbf_train_step(self,
                         state: TrainState,
                         batch: T.Dict[str, T.Any],
                         step: int) -> TrainState:
        def loss_fn(params):
            V_func = lambda x: state.apply_fn({"params": params}, x)

            # term 1: goal position
            loss1_b = V_func(batch["goal_states"])
            loss1 = jnp.mean(loss1_b)

            def loss234_single(x):
                sol, (V, Vdot) = self.solve_CLBF_QP(V_func, x)

                # apply descent loss to safe regions only
                safe_active = nn.sigmoid(10 * (self.safe_level + 0.1 - lax.stop_gradient(V)))

                # term 2: relaxation
                success = (sol.state.status == BoxOSQP.SOLVED)
                loss2 = lax.select(success, sol.params.primal[-1], 0.)
                loss2 *= safe_active

                # term 3: linearization
                loss3 = lax.select(success, nn.relu(1.0 + Vdot + self.clf_lambda * V), 0.)
                loss3 *= safe_active

                # term 4: simulation
                u = sol.params.primal[:-1]
                x_next = x + self.dt * self.dynamics.dynamics(x, u)
                V_next = V_func(x_next)
                loss4 = lax.select(success, nn.relu(1.0 + (V_next - V) / self.dt + self.clf_lambda * V), 0.)
                loss4 *= safe_active

                # term 5 safe / unsafe
                is_safe = self.dynamics.safety_mask(x)
                safe_violation = nn.relu(0.01 + V - self.safe_level)
                unsafe_violation = nn.relu(0.01 + self.safe_level - V)
                loss5 = lax.select(is_safe, safe_violation, unsafe_violation)

                return loss2, loss3, loss4, loss5, success.astype(float)

            loss2_b, loss3_b, loss4_b, loss5_b, success_b = jax.vmap(loss234_single)(batch["rand_states"])
            loss2 = jnp.sum(loss2_b) / jnp.sum(success_b)
            loss3 = jnp.sum(loss3_b) / jnp.sum(success_b)
            loss4 = jnp.sum(loss4_b) / jnp.sum(success_b)
            loss5 = jnp.mean(loss5_b)

            loss = 1e1 * loss1 + 1e2 * loss2 + loss3 + loss4 + loss5

            return loss, {
                "loss": loss,
                "loss1": loss1,
                "loss2": loss2,
                "loss3": loss3,
                "loss4": loss4,
                "loss5": loss5,
            }

        grads, losses = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, losses

    def train_step(self, batch: T.Dict[str, T.Any], step: int) -> T.Dict[str, T.Any]:
        self.state, metadict = self._clbf_train_step(self.state, batch, step)
        return metadict

