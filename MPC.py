import numpy as np
from world import World
from entities import Car
from geometry import Point
import copy
import multiprocessing as mp
import time



class MPC:
    def __init__(self, world, agent_idx, dest, gamma=0.97):
        self.world = world
        self.agent_idx = agent_idx
        self.dest = dest
        self.dest_radius = 5
        self.collision_penalty = -10000
        self.arrival_reward = 1000
        self.control_penalty = 1
        self.steering = [-0.3, 0.3]
        self.acc = [-0.5, 0, 0.5]
        self.steering_std = np.std(np.array(self.steering)) * 3
        self.acc_std = np.std(np.array(self.acc)) * 2
        self.d = 45
        self.m = 4
        self.gamma = gamma
        self.roll_out_policy_decay = 0.95

    def U_d(self, world):
        loc = world.dynamic_agents[self.agent_idx].center
        loc = np.array([loc.x, loc.y])
        return self.arrival_reward/np.linalg.norm(loc - self.dest)

    def R_sa(self, world, steer, acc):
        R = self.control_penalty * (abs(steer) + abs(acc) )
        agent = world.dynamic_agents[self.agent_idx]
        loc = agent.center
        loc = np.array([loc.x, loc.y])
        if np.linalg.norm(loc - self.dest) < self.dest_radius:
            R += self.arrival_reward
        l = len(world.dynamic_agents)
        for i in range(l):
            if i != self.agent_idx and agent.collidesWith(world.dynamic_agents[i]):
                R += self.collision_penalty
        return R

    def look_ahead_with_rollouts(self , act= None):

        def roll_out_policy(h_world):
            agent = h_world.dynamic_agents[self.agent_idx]
            return self.roll_out_policy_decay * agent.angular_velocity, self.roll_out_policy_decay * agent.acceleration

        def rollout(self, h_world, d, steer=None, acc=None):
            # print(f"car {self.agent_idx}")
            ret = 0
            for i in range(d):
                if steer is None or acc is None:
                    steer, acc = roll_out_policy(h_world)
                ret += self.R_sa(h_world,steer,acc) * self.gamma ** i
                for j, c in enumerate(h_world.dynamic_agents):
                    if j == self.agent_idx:
                        c.set_control(steer, acc)
                    else:
                        curr_acc = np.random.normal(0.0, self.acc_std)
                        curr_steer = np.random.normal(0.0, self.steering_std)
                        c.set_control(curr_steer, curr_acc)
                h_world.tick()
                # h_world.render()
                # time.sleep(dt / 8)
                steer, acc = None, None
            ret += self.U_d(h_world) * self.gamma ** self.m
            return ret

        Q = []
        action = []
        for steer in self.steering:
            for acc in self.acc:
                sum = 0
                for i in range(self.m):
                    h_world = copy.deepcopy(self.world)
                    sum += rollout(self, h_world, self.d, steer, acc)
                Q.append(sum / float(self.m))
                action.append((steer, acc))
        Q = np.array(Q)
        idx = np.argmax(Q)
        if act != None:
            act[self.agent_idx] = action[idx]
        return action[idx]




if __name__ == "__main__":
    dt = 0.2
    w = World(dt, width=120, height=120, ppm=6)
    renderer = copy.deepcopy(w)

    c0 = Car(Point(40, 20),np.pi)
    c0.velocity = Point(3.0, 0)
    w.add(c0)
    controller0 = MPC(w, 0, np.array([10, 20]))
    renderer.add(c0)

    c1 = Car(Point(10, 20), 0, 'blue')
    c1.velocity = Point(3.0, 0)
    w.add(c1)
    renderer.add(c1)
    controller1 = MPC(w, 1, np.array([40, 20]))

    c2 = Car(Point(25, 5), np.pi/2, 'green')
    c2.velocity = Point(3.0, 0)
    w.add(c2)
    renderer.add(c2)
    controller2 = MPC(w, 2, np.array([25, 40]))

    c3 = Car(Point(25, 40), -np.pi/2, 'purple')
    c3.velocity = Point(3.0, 0)
    w.add(c3)
    renderer.add(c3)
    controller3 = MPC(w, 3, np.array([25, 5]))

    c4 = Car(Point(40, 25), np.pi, 'black')
    c4.velocity = Point(3.0, 0)
    w.add(c4)
    controller4 = MPC(w, 4, np.array([10, 25]))
    renderer.add(c4)

    c5 = Car(Point(10, 25), 0, 'yellow')
    c5.velocity = Point(3.0, 0)
    w.add(c5)
    renderer.add(c5)
    controller5 = MPC(w, 5, np.array([40, 25]))

    for i in range(200):
        start = time.time()
        manager = mp.Manager()
        act = manager.dict()
        p0 = mp.Process(target=controller0.look_ahead_with_rollouts, args= (act, ))
        p1 = mp.Process(target=controller1.look_ahead_with_rollouts, args=(act,))
        p2 = mp.Process(target=controller2.look_ahead_with_rollouts, args=(act,))
        p3 = mp.Process(target=controller3.look_ahead_with_rollouts, args=(act,))
        p4 = mp.Process(target=controller4.look_ahead_with_rollouts, args=(act,))
        p5 = mp.Process(target=controller5.look_ahead_with_rollouts, args=(act,))
        p0.start()
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p0.join()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        c0.set_control(*act[0])
        c1.set_control(*act[1])
        c2.set_control(*act[2])
        c3.set_control(*act[3])
        c4.set_control(*act[4])
        c5.set_control(*act[5])
        end = time.time()
        print(end-start)
        w.tick()

        renderer.render()

        if w.collision_exists():
            print('Collision exists somewhere...')

