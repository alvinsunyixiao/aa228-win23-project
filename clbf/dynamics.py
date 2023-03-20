import typing as T

from abc import ABC, abstractmethod

import jax
import numpy as np

from jax import numpy as jnp
from jax import lax

##################################################################
# Dynamical Model
#   states = [x, y, theta]
#   controls = [v, w]
##################################################################

class AffineCtrlSys(ABC):
    @abstractmethod
    def f(self, states):
        pass

    @abstractmethod
    def g(self, states):
        pass

    def control_limit(self):
        return -jnp.inf * jnp.ones(self.control_dim), jnp.inf * jnp.ones(self.control_dim)

    def state_limit(self):
        return -jnp.inf * jnp.ones(self.state_dim), jnp.inf * jnp.ones(self.state_dim)

    @property
    @abstractmethod
    def state_dim(self) -> int:
        return 0

    @property
    @abstractmethod
    def control_dim(self) -> int:
        return 0

    def dynamics(self, states, controls):
        return self.f(states) + (self.g(states) @ controls[..., jnp.newaxis])[..., 0]

    def u_nominal(self, states):
        return jnp.zeros(self.control_dim)

    def h(self, states, *args):
        return 0.

class ExtendedUnicycle(AffineCtrlSys):
    def __init__(self,
        control_limit: T.Sequence[float] = (2., np.pi),
        safety_thresh: float = 0.5,
        active_thresh: float = 2.,
    ):
        self.ulim = jnp.array(control_limit)
        self.safety_thresh = safety_thresh
        self.active_thresh = active_thresh
        self.rng = jax.random.PRNGKey(42)

    def f(self, states):
        states_dot = jnp.zeros_like(states)
        v = states[..., 2]
        theta = states[..., 3]

        states_dot = states_dot.at[..., 0].set(v * jnp.cos(theta))
        states_dot = states_dot.at[..., 1].set(v * jnp.sin(theta))

        return states_dot

    def g(self, states):
        theta = states[..., 3]

        x_zeros = jnp.zeros_like(theta)
        x_ones = jnp.ones_like(theta)

        col1 = jnp.stack([x_zeros, x_zeros, x_ones, x_zeros], axis=-1)
        col2 = jnp.stack([x_zeros, x_zeros, x_zeros, x_ones], axis=-1)

        return jnp.stack([col1, col2], axis=-1)

    def control_limit(self):
        return -self.ulim, self.ulim

    def u_nominal(self, states):
        x = states[..., 0]
        y = states[..., 1]
        v = states[..., 2]
        theta = states[..., 3]

        rho = jnp.linalg.norm(states[..., :2], axis=-1)
        bearing = lax.select(rho > 1e-3, jnp.arctan2(-y, -x), theta)

        k1 = 0.5
        k2 = 1.6
        k3 = 1.0

        def wrap(ang):
            return (ang + jnp.pi) % (2 * jnp.pi) - jnp.pi

        a = k1 * rho * jnp.cos(theta - bearing) - k2 * v
        t1 = wrap(bearing - theta)
        t2 = wrap(bearing - theta + jnp.pi)
        w = k3 * lax.select(jnp.abs(t1) <= jnp.abs(t2), t1, t2)

        a = jnp.clip(a, -self.ulim[0], self.ulim[0])
        w = jnp.clip(w, -self.ulim[1], self.ulim[1])

        return jnp.stack([a, w], axis=-1)

    def h(self, states, obstacles):
        xy_b2 = states[..., :2]
        v_b1 = states[..., 2, jnp.newaxis]
        theta_b1 = states[..., 3, jnp.newaxis]
        obs_xy_b2 = obstacles[..., :2]
        obs_vel_b2 = obstacles[..., 2:]

        vel_xy_b2 = jnp.concatenate([jnp.cos(theta_b1), jnp.sin(theta_b1)], axis=-1) * v_b1
        vel_local_b2 = vel_xy_b2 - obs_vel_b2
        vel_mag_b = jnp.sqrt(jnp.sum(vel_local_b2**2, axis=-1) + 1e-8)

        xy_local_b2 = obs_xy_b2 - xy_b2
        dists_b = jnp.sqrt(jnp.sum(xy_local_b2**2, axis=-1))
        cos_phi_b = jnp.sqrt(dists_b**2 - self.safety_thresh**2) / dists_b

        h = jnp.sum(xy_local_b2 * vel_local_b2, axis=-1) / dists_b - vel_mag_b * cos_phi_b

        return lax.select(dists_b < self.active_thresh, h, jnp.zeros_like(h))

    @property
    def state_dim(self):
        return 4

    @property
    def control_dim(self):
        return 2


class Unicycle(AffineCtrlSys):
    def __init__(self,
        control_limit: T.Sequence[float] = (2., np.pi),
        safety_thresh: float = 0.5,
    ):
        self.ulim = jnp.array(control_limit)
        self.safety_thresh = safety_thresh

    def f(self, states):
        return jnp.zeros_like(states)

    def g(self, states):
        theta = states[..., 2]

        x_zeros = jnp.zeros_like(theta)
        x_ones = jnp.ones_like(theta)

        col1 = jnp.stack([jnp.cos(theta), jnp.sin(theta), x_zeros], axis=-1)
        col2 = jnp.stack([x_zeros, x_zeros, x_ones], axis=-1)

        return jnp.stack([col1, col2], axis=-1)

    def control_limit(self):
        return -self.ulim, self.ulim

    def u_nominal(self, states):
        x = states[..., 0]
        y = states[..., 1]
        theta = states[..., 2]

        rho = jnp.linalg.norm(states[..., :2], axis=-1)
        bearing = lax.select(rho > 1e-3, jnp.arctan2(-y, -x), theta)

        k1 = 0.2
        k2 = 1.2

        def wrap(ang):
            return (ang + jnp.pi) % (2 * jnp.pi) - jnp.pi

        v = k1 * rho * jnp.cos(theta - bearing)
        t1 = wrap(bearing - theta)
        t2 = wrap(bearing - theta + jnp.pi)
        w = k2 * lax.select(jnp.abs(t1) <= jnp.abs(t2), t1, t2)

        return jnp.stack([v, w], axis=-1)

    def h(self, states, obstacles):
        xy_b2 = states[..., :2]

        xy_local_b2 = obstacles - xy_b2
        dists_square_b = jnp.sum(xy_local_b2**2, axis=-1)

        return self.safety_thresh**2 - dists_square_b

    @property
    def state_dim(self):
        return 3

    @property
    def control_dim(self):
        return 2

