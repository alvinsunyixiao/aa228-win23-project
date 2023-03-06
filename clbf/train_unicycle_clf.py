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
