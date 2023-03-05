import jax.numpy as np
from geometry import Point, Rectangle, Circle, Ring
from typing import Union
import copy


class Entity:
    def __init__(self, center: Point, heading: float, movable: bool = True, friction: float = 0):
        self.center = center # this is x, y
        self.heading = heading
        self.movable = movable
        self.color = 'ghost white'
        self.collidable = True
        if movable:
            self.friction = friction
            self.velocity = Point(0,0) # this is xp, yp
            self.acceleration = 0 # this is vp (or speedp)
            self.inputAcceleration = 0
            self.angular_velocity = 0
            self.max_speed = np.inf
            self.min_speed = 0
    
    @property
    def speed(self) -> float:
        return self.velocity.norm(p = 2) if self.movable else 0
    
    def set_control(self, angular_velocity: float, inputAcceleration: float ):
        self.inputAcceleration = inputAcceleration
        self.angular_velocity = angular_velocity
    
    @property
    def rear_dist(self) -> float: # distance between the rear wheels and the center of mass. This is needed to implement the kinematic bicycle model dynamics
        if isinstance(self, RectangleEntity):
            # only for this function, we assume
            # (i) the longer side of the rectangle is always the nominal direction of the car
            # (ii) the center of mass is the same as the geometric center of the RectangleEntity.
            return np.maximum(self.size.x, self.size.y) / 2.
        raise NotImplementedError
    
    def tick(self, dt: float):
        if self.movable:
            speed = self.speed
            heading = self.heading
            angular_velocity= self.angular_velocity
            
            new_acceleration = self.inputAcceleration - self.friction
            new_speed = np.clip(speed + new_acceleration * dt, self.min_speed, self.max_speed)
            new_heading = heading + angular_velocity*dt
            angle = (heading + new_heading)/2.
            new_center = self.center + (speed + new_speed)*Point(np.cos(angle), np.sin(angle))*dt / 2.
            new_velocity = Point(new_speed * np.cos(new_heading), new_speed * np.sin(new_heading))
            
            self.center = new_center
            self.heading = np.mod(new_heading, 2*np.pi)
            self.velocity = new_velocity
            self.acceleration = new_acceleration
            self.buildGeometry()
    
    def buildGeometry(self): # builds the obj
        raise NotImplementedError
        
    def collidesWith(self, other: Union['Point','Entity']) -> bool:
        if isinstance(other, Entity):
            return self.obj.intersectsWith(other.obj)
        elif isinstance(other, Point):
            return self.obj.intersectsWith(other)
        raise NotImplementedError
        
    def distanceTo(self, other: Union['Point','Entity']) -> float:
        if isinstance(other, Entity):
            return self.obj.distanceTo(other.obj)
        elif isinstance(other, Point):
            return self.obj.distanceTo(other)
        raise NotImplementedError
        
    def copy(self):
        return copy.deepcopy(self)
        
    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y
        
    @property
    def xp(self):
        return self.velocity.x

    @property
    def yp(self):
        return self.velocity.y
    
class RectangleEntity(Entity):
    def __init__(self, center: Point, heading: float, size: Point, movable: bool = True, friction: float = 0):
        super(RectangleEntity, self).__init__(center, heading, movable, friction)
        self.size = size
        self.buildGeometry()
    
    @property
    def edge_centers(self):
        x = self.center.x
        y = self.center.y
        w = self.size.x
        h = self.size.y
        edge_centers = np.array([[x + w / 2. * np.cos(self.heading), y + w / 2. * np.sin(self.heading)],
        [x - h / 2. * np.sin(self.heading), y + h / 2. * np.cos(self.heading)],
        [x - w / 2. * np.cos(self.heading), y - w / 2. * np.sin(self.heading)],
        [x + h / 2. * np.sin(self.heading), y - h / 2. * np.cos(self.heading)]])
        return edge_centers
        
    @property
    def corners(self):
        ec = self.edge_centers
        c = np.array([self.center.x, self.center.y])
        corners = []
        corners.append(Point(*(ec[1] + ec[0] - c)))
        corners.append(Point(*(ec[2] + ec[1] - c)))
        corners.append(Point(*(ec[3] + ec[2] - c)))
        corners.append(Point(*(ec[0] + ec[3] - c)))
        return corners
        
    def buildGeometry(self):
        C = self.corners
        self.obj = Rectangle(*C[:-1])


class Car(RectangleEntity):
    def __init__(self, center: Point, heading: float, color: str = 'red'):
        size = Point(4., 2.)
        movable = True
        friction = 0.06
        super(Car, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = True