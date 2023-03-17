import numpy as np
from world import World
from entities import Car
from geometry import Point
import copy
import multiprocessing as mp
import time



class MPC:
    def __init__(self, world, agent_idx, dest, gamma=0.94):
        self.world = world
        self.agent_idx = agent_idx
        self.dest = dest
        self.dest_radius = 5
        self.collision_penalty = -10000
        self.arrival_reward = 1000
        self.control_penalty = 0
        self.velocity_penalty = -0.05
        self.steering = [-0.3, 0.3]
        self.acc = [-0.5, 0, 0.5]
        self.steering_std = np.std(np.array(self.steering)) * 4
        self.acc_std = np.std(np.array(self.acc)) * 3
        self.d = 30
        self.m = 6
        self.gamma = gamma
        self.roll_out_policy_decay = 0.90

    def U_d(self, world):
        loc = world.dynamic_agents[self.agent_idx].center
        loc = np.array([loc.x, loc.y])
        return self.arrival_reward/np.linalg.norm(loc - self.dest)

    def R_sa(self, world, steer, acc):
        agent = world.dynamic_agents[self.agent_idx]
        R =  self.control_penalty * (abs(steer) + abs(acc)) + self.velocity_penalty * agent.velocity.norm(p = 2)
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
                #h_world.render()
                #time.sleep(dt / 8)
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

def set_up_car(w, renderer, controllers, cars, start, start_angle, dest, color='red', initial_velo = 3):
    c= Car(Point(*start),start_angle, color)
    cars.append(c)
    c.velocity = Point(initial_velo, 0)
    w.add(c)
    controllers.append(MPC(w, len(controllers), np.array(dest)))
    renderer.add(c)


if __name__ == "__main__":
    dt = 0.2
    w = World(dt, width=120, height=120, ppm=6)
    renderer = copy.deepcopy(w)
    controllers = []
    cars = []

    set_up_car(w,renderer,controllers, cars, (40,20), np.pi,(10,20))
    set_up_car(w, renderer, controllers, cars,(10, 20), 0, (40, 20))


    set_up_car(w,renderer,controllers,cars, (25,5), np.pi/2,(25,35), "purple")
    set_up_car(w, renderer, controllers, cars,(25, 35), -np.pi/2, (25, 5), "purple")

    for i in range(200):
        start = time.time()
        manager = mp.Manager()
        act = manager.dict()
        l = len(cars)
        ps = []
        for i in range(l):
            p = mp.Process(target=controllers[i].look_ahead_with_rollouts, args=(act,))
            p.start()
            ps.append(p)
        for i in range(l):
            ps[i].join()
            cars[i].set_control(*act[i])
        end = time.time()
        #print(end-start)
        w.tick()

        renderer.render()

        if w.collision_exists():
            print('Collision exists somewhere...')

