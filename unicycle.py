import jax.numpy as jnp

##################################################################
# Dynamical Model
#   states = [x, y, theta]
#   controls = [v, w]
##################################################################

def f(states):
    return jnp.zeros_like(states)

def g(states):
    theta = states[..., 2]

    col1 = jnp.stack([jnp.cos(theta), jnp.sin(theta), jnp.zeros_like(theta)], axis=-1)
    col2 = jnp.stack([jnp.zeros_like(theta), jnp.zeros_like(theta), jnp.ones_like(theta)], axis=-1)

    return jnp.stack([col1, col2], axis=-1)

def dynamics(states, controls):
    return f(states) + (g(states) @ controls[..., jnp.newaxis])[..., 0]
