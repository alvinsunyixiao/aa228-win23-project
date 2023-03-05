import jax
import jax.numpy as np
from world import World
from entities import Car
from geometry import Point
import time

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(20,20), np.pi/2)
w.add(c1)

c2 = Car(Point(118,90), np.pi, 'blue')
c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
w.add(c2)

w.render() # This visualizes the world we just constructed.


# Let's implement some simple scenario with all agents
c1.set_control(0, 0.35)
c2.set_control(0, 0.05)
for k in range(400):
    # All movable objects will keep their control the same as long as we don't change it.
    if k == 100: # Let's say the first Car will release throttle (and start slowing down due to friction)
        c1.set_control(0, 0)
    elif k == 200: # The first Car starts pushing the brake a little bit. The second Car starts turning right with some throttle.
        c1.set_control(0, -0.02)
    elif k == 325:
        c1.set_control(0, 0.8)
        c2.set_control(-0.45, 0.3)
    elif k == 367: # The second Car stops turning.
        c2.set_control(0, 0.1)
    w.tick() # This ticks the world for one time step (dt second)
    w.render()
    time.sleep(dt/4) # Let's watch it 4x
    if w.collision_exists(): # Or we can check if there is any collision at all.
        print('Collision exists somewhere...')
w.close()
