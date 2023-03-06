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

from dynamics import Unicycle
from model import CLBF

if __name__ == "__main__":
    dynamics = Unicycle(3)
    clbf = CLBF(dynamics)

    for i in range(10000):
        batch = {
            "goal_states": dynamics.random_states(),
            "rand_states": dynamics.random_goal_states(),
        }
        losses = clbf.train_step(batch)
        print(losses["loss1"], losses["loss2"], losses["loss3"])

        if i % 100 == 0:
            clbf.save("ckpts", step=i)
