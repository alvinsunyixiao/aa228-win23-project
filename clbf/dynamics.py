import typing as T

from abc import ABC, abstractmethod

import jax
import numpy as np
import jax.numpy as jnp

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

    @abstractmethod
    def safety_mask(self, states):
        pass

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


class Freeflyer(AffineCtrlSys):
    def __init__(self,
        num_neighbors: int = 1,
        state_limit: T.Sequence[float] = (2., 2., 1., 1.),
        control_limit: T.Sequence[float] = (5., 5.),
        safety_thresh: float = 0.2,
    ):
        self.num_neighbors = num_neighbors
        self.safety_thresh = safety_thresh
        self.xlim = jnp.array(state_limit)
        self.ulim = jnp.array(control_limit)
        self.rng = jax.random.PRNGKey(42)

    def f(self, states):
        return jnp.zeros_like(states).at[..., :2].set(states[..., 2:4])

    def g(self, states):
        x_zeros = jnp.zeros_like(states[..., 0])
        x_ones = jnp.ones_like(states[..., 0])

        col1 = jnp.stack([x_zeros, x_zeros, x_ones, x_zeros], axis=-1)
        col2 = jnp.stack([x_zeros, x_zeros, x_zeros, x_ones], axis=-1)

        g_raw = jnp.stack([col1, col2], axis=-1)
        g_obs = jnp.zeros(g_raw.shape[:-2] + (self.num_neighbors * 2, 2))

        return jnp.concatenate([g_raw, g_obs], axis=-2)

    def control_limit(self):
        return -self.ulim, self.ulim

    def state_is_safe(self, states):
        return jnp.all((states[..., :4] >= -self.xlim) & (states[..., :4] <= self.xlim), axis=-1)

    def safety_mask(self, states):
        state_mask = self.state_is_safe(states)
        if self.num_neighbors == 0:
            return state_mask

        neighbors = states[..., 4:]
        neighbors_bk2 = jnp.reshape(neighbors, states.shape[:-2] + (-1, 2))
        self_b2 = states[..., :2]
        self_b12 = self_b2[..., jnp.newaxis, :]

        dists_bk = jnp.linalg.norm(self_b12 - neighbors_bk2, axis=-1)
        min_dists_b = jnp.min(dists_bk, axis=-1)

        return (min_dists_b > self.safety_thresh) & state_mask

    def random_states(self,
        batch_shape: T.Union[int, T.Tuple[int, ...]] = 2**12,
        max_obs_dist: float = 4.,
    ):
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        self.rng, subrng = jax.random.split(self.rng)
        eps = 0.5
        self_b4 = jax.random.uniform(self.rng, batch_shape + (4,), minval=-self.xlim * (1 + eps), maxval=self.xlim * (1 + eps))
        self_xy = self_b4[..., :2]

        others_dir_bk2 = np.random.normal(size=batch_shape + (self.num_neighbors, 2))
        others_dir_bk2 /= np.maximum(np.linalg.norm(others_dir_bk2, axis=-1, keepdims=True), 1e-9)
        others_dist_bk1 = np.random.uniform(0, max_obs_dist, size=batch_shape + (self.num_neighbors, 1))
        others_xy_bk2 = self_xy[..., jnp.newaxis, :] + others_dir_bk2 * others_dist_bk1
        others_xy_bl = np.reshape(others_xy_bk2, batch_shape + (-1,))

        return jnp.concatenate([self_b4, others_xy_bl], axis=-1)

    def random_goal_states(self,
        batch_shape: T.Union[int, T.Tuple[int, ...]] = 2**12,
        max_obs_dist: float = 2.,
    ):
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        self_b4 = np.zeros(batch_shape + (4,))

        others_dir_bk2 = np.random.normal(size=batch_shape + (self.num_neighbors, 2))
        others_dir_bk2 /= np.maximum(np.linalg.norm(others_dir_bk2, axis=-1, keepdims=True), 1e-9)
        others_dist_bk1 = np.random.uniform(0, max_obs_dist, size=batch_shape + (self.num_neighbors, 1))
        others_xy_bk2 = others_dir_bk2 * others_dist_bk1
        others_xy_bl = np.reshape(others_xy_bk2, batch_shape + (-1,))

        return np.concatenate([self_b4, others_xy_bl], axis=-1)

    @property
    def state_dim(self):
        return 4 + self.num_neighbors * 2

    @property
    def control_dim(self):
        return 2

