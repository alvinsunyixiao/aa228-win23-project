from world import World
from entities import Car
from geometry import Point
from model import Simple_MLP
import time
import torch
import random
import math
dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.

#### loss function ####
def loss_fn(agent_A, agent_B, dest_A, out):
    #out = torch.maximum(out, torch.tensor([-5.0]))
    #out = torch.minimum(out, torch.tensor([5.0]))
    out = out - 0.5
    new_heading = agent_A.heading + out * dt
    angle = (agent_A.heading + new_heading) / 2.
    new_Ax = agent_A.center.x + agent_A.speed * torch.cos(angle) * dt / 2.
    new_Ay = agent_A.center.y + agent_A.speed * torch.sin(angle) * dt / 2.
    
    return ((new_Ax - dest_A.x) ** 2 + (new_Ay - dest_A.y) ** 2)**(1./2)# - ((new_Ax - agent_A.center.x) ** 2 + (new_Ay - agent_B.center.y) ** 2)**(1./2)

#### Training ####
model = Simple_MLP(num_feature=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

num_epoch = 100
for i in range(num_epoch):
    # World #
    human_controller = False
    w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

    # Car 1 #
    #randO = random.uniform(0, 2)
    #pi = torch.tensor(randO * math.pi)
    pi = torch.tensor(math.pi / 2)
    randX = float(random.randint(55, 65))
    randY = float(random.randint(5, 15))
    print(f"Epoch {i}")
    print(f"Start: {randX}, {randY}, {pi}*pi")
    c1 = Car(Point(randX, randY), pi) # center_location, heading, color
    c1.set_control(torch.tensor(0.), 0.) # angular_velocity, acceleration
    c1.velocity = Point(4., 0.) # init_velocity: (x, y)
    randDX = float(random.randint(55, 65))
    randDY = float(random.randint(105, 115))
    print("Dest: ", randDX, randDY)
    print()
    dest1 = Point(randDX, randDY) # destination for c1
    w.add(c1)

    # Car 2 #
    c2 = Car(Point(118., 60.), pi, 'blue') # center_location, heading, color
    c2.set_control(torch.tensor(0.), 0.) # angular_velocity, acceleration
    c2.velocity = Point(0., 0.) # init_velocity: (x, y)
    dest2 = Point(0., 60.) # destination for c2
    w.add(c2)
    w.render() # This visualizes the world we just constructed.

    for t in range(300):
        x = torch.zeros([1,5], dtype=torch.float32)
        x[0,0] = c1.center.x
        x[0,1] = c1.center.y
        x[0,2] = c1.heading.item()
        #x[0,3] = c1.speed
        #x[0,4] = c1.angular_velocity
        x[0,3] = dest1.x
        x[0,4] = dest1.y
        #x[0,7] = c1.distanceTo(dest1)
        #x[0,4] = c1.acceleration
        #x[0,7] = c2.center.x
        #x[0,8] = c2.center.y
        #x[0,9] = c2.heading.item()
        #x[0,10] = c2.speed
        #x[0,11] = c2.acceleration
        #x[0,12] = c2.angular_velocity
        #x[0,13] = c1.distanceTo(c2)

        out = model(x)
        #c1.set_control(out[0,0].item(), 0.)
        loss = loss_fn(c1, c2, dest1, out)
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            #out = torch.maximum(out, torch.tensor([-5.0]))
            #out = torch.minimum(out, torch.tensor([5.0]))
            out = out - 0.5
            print(out)
            c1.set_control(out[0,0], 0.)
            w.tick() # This ticks the world for one time step (dt second)
            w.render()
            time.sleep(dt/2) # Let's watch it 2x
            if c1.distanceTo(dest1) < 4.0:
                break

    w.close()

torch.save(model.state_dict(), "Simple_MLP_1.pt")