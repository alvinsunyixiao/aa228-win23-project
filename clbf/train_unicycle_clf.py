import argparse

from jax.config import config

from dynamics import Unicycle
from model import CLBF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="ckpts",
                        help="output directory to store checkpoints")

    return parser.parse_args()

if __name__ == "__main__":
    #config.update("jax_enable_x64", True)

    args = parse_args()

    dynamics = Unicycle(0)
    clbf = CLBF(dynamics)

    for i in range(10000):
        batch = {
            "rand_states": dynamics.random_states(),
            "goal_states": dynamics.random_goal_states(),
        }
        losses = clbf.train_step(batch, i)
        print(losses["loss"], losses["loss1"], losses["loss2"], losses["loss3"], losses["loss4"])

        if i % 100 == 0:
            clbf.save(args.output, step=i)
