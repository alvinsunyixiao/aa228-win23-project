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
from flax.training.checkpoints import save_checkpoint

from unicycle import Unicycle

dynamics = Unicycle()

class CLBF(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=48)(x)
        x = nn.elu(x)
        x = nn.Dense(features=48)(x)
        x = nn.elu(x)
        x = nn.Dense(features=8)(x)
        return jnp.sum(x * x, axis=-1)

@jax.jit
def train_step(state: TrainState, batch: T.Dict[str, T.Any]) -> TrainState:
    def loss_fn(params):
        V_func = lambda x: state.apply_fn({"params": params}, x)

        # term 1: goal position
        loss1_b = V_func(batch["goal_states"])

        # term 2: feasibility
        def loss23_single(x):
            V, V_D_x_s = jax.value_and_grad(V_func)(x)

            Lf_V = V_D_x_s @ dynamics.f(x)
            Lg_V_c = V_D_x_s @ dynamics.g(x)

            qp = OSQP()
            ur_dim = dynamics.control_dim + 1

            # objective
            Q = jnp.eye(ur_dim)
            Q = Q.at[2, 2].set(0.)
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
            r = lax.select(sol.state.status == BoxOSQP.SOLVED, sol.params.primal[-1], 0.)
            u = sol.params.primal[:-1]

            # linearization loss
            Vdot = Lf_V + Lg_V_c @ u

            return r, nn.relu(1.0 + Vdot + V)

        loss2_b, loss3_b = jax.vmap(loss23_single)(batch["rand_states"])

        loss_b = 1e1 * loss1_b + loss2_b + loss3_b
        return jnp.mean(loss_b), {"loss1": jnp.mean(loss1_b),
                                  "loss2": jnp.mean(loss2_b),
                                  "loss3": jnp.mean(loss3_b)}

    grads, losses = jax.grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, losses


if __name__ == "__main__":
    clbf = CLBF()

    # init
    rng = jax.random.PRNGKey(0)
    params = clbf.init(rng, jnp.ones((1, dynamics.state_dim)))["params"]
    tx = optax.sgd(1e-3, 0.9)
    state = TrainState.create(apply_fn=clbf.apply, params=params, tx=tx)

    # checkpointer
    orbax_ckpter = orbax.checkpoint.PyTreeCheckpointer()

    for i in range(10000):
        batch = {
            "goal_states": dynamics.random_states(),
            "rand_states": dynamics.random_goal_states(),
        }
        state, losses = train_step(state, batch)
        print(losses["loss1"], losses["loss2"], losses["loss3"])
        if i % 100 == 0:
            save_checkpoint(ckpt_dir="ckpts",
                            target=state,
                            step=i,
                            keep=10,
                            orbax_checkpointer=orbax_ckpter)
