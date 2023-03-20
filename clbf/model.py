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


class CBF:
    def __init__(self,
        dynamics: AffineCtrlSys,
        cbf_lambda: float = 1.,
        dt: float = 1e-2,
        mlp_configs: T.Tuple[int, ...] = (64, 64),
    ) -> None:
        self.dynamics = dynamics
        self.cbf_lambda = cbf_lambda
        self.dt = dt

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(self, state, obstacles, Q=1., u_ref=None, u_init=None):
        sol, (h, hdot) = self.solve_CBF_QP(state, obstacles, Q=Q, u_ref=u_ref, u_init=u_init, maxiter=100000)
        return sol.params.primal, (h, hdot, sol)

    def solve_CBF_QP(self, state, obstacles, Q=1., u_ref=None, u_init=None, maxiter=1000):
        h = []
        h_D_x = []
        for i in range(obstacles.shape[0]):
            h_func = lambda x: self.dynamics.h(x, obstacles[i])
            h_tmp, h_D_x_tmp = jax.value_and_grad(h_func)(state)
            h.append(h_tmp)
            h_D_x.append(h_D_x_tmp)

        h = jnp.stack(h)
        h_D_x = jnp.stack(h_D_x)

        Lf_h = h_D_x @ self.dynamics.f(state)
        Lg_h = h_D_x @ self.dynamics.g(state)

        if isinstance(Q, float):
            Q = Q * jnp.ones(self.dynamics.control_dim)

        Q_mat = 2. * jnp.diag(Q)
        if u_ref is None:
            u_ref = self.dynamics.u_nominal(state)
        c = -2. * Q * u_ref

        G = jnp.concatenate([Lg_h, -jnp.eye(self.dynamics.control_dim), jnp.eye(self.dynamics.control_dim)], axis=0)
        b_cbf = -self.cbf_lambda*h - Lf_h
        ulim_low, ulim_high = self.dynamics.control_limit()
        b = jnp.concatenate([b_cbf, -ulim_low, ulim_high])

        # solve OSQP
        if u_init is None:
            u_init = u_ref
        qp = OSQP(eq_qp_solve="cg+jacobi", maxiter=maxiter)
        init_params = qp.init_params(u_init, (Q_mat, c), None, (G, b))
        sol = qp.run(init_params, params_obj=(Q_mat, c), params_ineq=(G, b))

        hdot = Lf_h + Lg_h @ sol.params.primal

        return sol, (h, hdot)

