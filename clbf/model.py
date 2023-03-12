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
    u_dim: int
    mlp_configs: T.Tuple[int, ...] = (64, 64)
    cert_dim: int = 8

    extend: T.Callable = lambda x: x
    prior: T.Callable = lambda x: 0

    @nn.compact
    def __call__(self, x):
        prior = self.prior(x)

        x = self.extend(x)
        for i, feat in enumerate(self.mlp_configs):
            x = nn.Dense(features=feat)(x)
            x = nn.tanh(x)

        Vh = nn.Dense(features=self.cert_dim)(x)
        V = 0.5 * jnp.sum(Vh * Vh, axis=-1) + prior

        u = nn.Dense(features=self.u_dim)(x)

        return V, u


class CLBF:
    def __init__(self,
        dynamics: AffineCtrlSys,
        clf_lambda: float = 1.,
        dt: float = 1e-2,
        mlp_configs: T.Tuple[int, ...] = (64, 64),
    ) -> None:
        self.dynamics = dynamics
        self.clf_lambda = clf_lambda
        self.dt = dt

        # initialize mlp
        self.mlp_cert = MLPCertificate(u_dim=self.dynamics.control_dim, mlp_configs=mlp_configs, extend=self.dynamics.extend)
        rng = jax.random.PRNGKey(0)
        params = self.mlp_cert.init(rng, jnp.ones((1, self.dynamics.state_dim)))["params"]
        tx = optax.sgd(optax.exponential_decay(1e-2, 4000, 0.1), 0.9)
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
        return sol.params.primal[:-1], sol.params.primal[-1]

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
            loss1_b, _ = V_func(batch["goal_states"])
            loss1 = jnp.mean(loss1_b)

            def loss23_single(x):
                (V, u), V_D_x_s = jax.value_and_grad(V_func, has_aux=True)(x)

                Lf_V = V_D_x_s @ self.dynamics.f(x)
                Lg_V_c = V_D_x_s @ self.dynamics.g(x)
                Vdot = Lf_V + Lg_V_c @ u

                # term 2: linearization
                loss2 = nn.relu(1.0 + Vdot + self.clf_lambda * V)

                # term 3: simulation
                x_next = x + self.dt * self.dynamics.dynamics(x, u)
                V_next, _ = V_func(x_next)
                loss3 = nn.relu(1.0 + (V_next - V) / self.dt + self.clf_lambda * V)

                loss4 = jnp.linalg.norm(u, axis=-1)

                return loss2, loss3, loss4

            loss2_b, loss3_b, loss4_b = jax.vmap(loss23_single)(batch["rand_states"])
            loss2 = jnp.mean(loss2_b)
            loss3 = jnp.mean(loss3_b)
            loss4 = jnp.mean(loss4_b)

            loss = 1e1 * loss1 + loss2 + loss3 + loss4

            return loss, {
                "loss": loss,
                "loss1": loss1,
                "loss2": loss2,
                "loss3": loss3,
                "loss4": loss4
            }

        grads, losses = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, losses

    def train_step(self, batch: T.Dict[str, T.Any], step: int) -> T.Dict[str, T.Any]:
        self.state, metadict = self._clbf_train_step(self.state, batch, step)
        return metadict

