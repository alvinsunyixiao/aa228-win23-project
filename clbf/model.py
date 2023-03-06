import functools
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import optax
import orbax
import typing as T

from jaxopt import OSQP, BoxOSQP
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from dynamics import AffineCtrlSys

class MLPCertificate(nn.Module):
    configs: T.Tuple[int, ...] = (48, 48, 8)

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.configs):
            x = nn.Dense(features=feat)(x)
            if i < len(self.configs) - 1:
                x = nn.elu(x)

        return jnp.sum(x * x, axis=-1)



class CLBF:
    def __init__(self, dynamics: AffineCtrlSys, mlp_configs: T.Sequence[int] = [48, 48, 8]):
        self.dynamics = dynamics

        # initialize mlp
        self.mlp_cert = MLPCertificate(configs=mlp_configs)
        rng = jax.random.PRNGKey(0)
        params = self.mlp_cert.init(rng, jnp.ones((1, self.dynamics.state_dim)))["params"]
        tx = optax.sgd(1e-3, 0.9)
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

    def solve_CLBF_QP(self, state, V_func: T.Callable):
        V, V_D_x_s = jax.value_and_grad(V_func)(state)

        Lf_V = V_D_x_s @ self.dynamics.f(state)
        Lg_V_c = V_D_x_s @ self.dynamics.g(state)

        qp = OSQP()
        ur_dim = self.dynamics.control_dim + 1

        # objective
        Q = jnp.eye(ur_dim)
        Q = Q.at[-1, -1].set(0.)
        c = jnp.zeros(ur_dim)
        c.at[-1].set(100)  # relaxation penalty

        # inequality
        G1 = jnp.concatenate([Lg_V_c, jnp.array([-1.])])
        h1 = -V - Lf_V
        G2 = jnp.zeros(ur_dim)
        G2 = G2.at[-1].set(-1.)
        h2 = 0.
        G = jnp.stack([G1, G2])
        h = jnp.stack([h1, h2])

        # solve OSQP
        sol = qp.run(params_obj=(Q, c), params_ineq=(G, h))
        Vdot = Lf_V + Lg_V_c @ sol.params.primal[:-1]

        return sol, (V, Vdot)

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(self, state):
        V_func = lambda x: self.mlp_cert.apply({"params": self.state.params}, x)

        qp_sol, (V, Vdot) = self.solve_CLBF_QP(state, V_func)

        return qp_sol.params.primal[:-1], (V, Vdot)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _clbf_train_step(self,
                         state: TrainState,
                         batch: T.Dict[str, T.Any]) -> TrainState:
        def loss_fn(params):
            V_func = lambda x: state.apply_fn({"params": params}, x)

            # term 1: goal position
            loss1_b = V_func(batch["goal_states"])

            # term 2: feasibility
            def loss23_single(x):
                V = V_func(x)
                qp_sol, (_, Vdot) = self.solve_CLBF_QP(x, V_func)

                r = lax.select(qp_sol.state.status == BoxOSQP.SOLVED, qp_sol.params.primal[-1], 0.)
                u = qp_sol.params.primal[:-1]

                return r, nn.relu(1.0 + Vdot + V)

            loss2_b, loss3_b = jax.vmap(loss23_single)(batch["rand_states"])

            loss_b = 1e1 * loss1_b + loss2_b + loss3_b
            return jnp.mean(loss_b), {"loss1": jnp.mean(loss1_b),
                                      "loss2": jnp.mean(loss2_b),
                                      "loss3": jnp.mean(loss3_b)}

        grads, losses = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, losses

    def train_step(self, batch: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        self.state, metadict = self._clbf_train_step(self.state, batch)
        return metadict

