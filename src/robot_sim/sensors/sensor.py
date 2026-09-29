from __future__ import annotations

from typing import Protocol

import numpy as np

from ..observation import ContactMeasurement, Measurement, PositionMeasurement, RangeMeasurement, RadialRangeArray
from ..robot import Robot
from ..world import World

import math

class Sensor(Protocol):
    def measure(self, robot: Robot, world: World) -> Measurement:
        ...


class ContactSensor:
    def __init__(self, radius: float):
        self.radius = radius

    def measure(self, robot: Robot, world: World) -> ContactMeasurement:
        for obstacle in world.obstacles:
            if obstacle.within(self.radius, robot.position):
                tangent = obstacle.tangent(robot.position, standoff=self.radius / 2)
                return ContactMeasurement(contact=True, tangent=tangent)
        return ContactMeasurement(contact=False, tangent=np.zeros(2))


class PositionSensor:
    def measure(self, robot: Robot, world: World) -> PositionMeasurement:
        # Copy so readings don't change as the robot moves
        return PositionMeasurement(robot.position.copy())

class RadialDistanceSensor:
    def __init__(self, resolution: int, max_range: float = 4.0, jump_threshold: float = 0.5):
        self.max_range = max_range
        self.jump_threshold = jump_threshold
        self.sensors = [DistanceSensor(2.0*math.pi/resolution*n, max_range) for n in range(resolution)]
    def measure(self, robot: Robot, world: World):
        measurements = [s.measure(robot, world) for s in self.sensors]
        
        return RadialRangeArray(measurements=measurements)
    



class DistanceSensor:
    def __init__(self, theta: float, max_range: float = 4.0):
        self.theta = theta
        self.max_range = max_range
    def measure(self, robot: Robot, world: World) -> RangeMeasurement:
        d = min((ob.intersect(robot.position, self.theta, self.max_range) for ob in world.obstacles), default=math.inf)
        return RangeMeasurement(theta=self.theta, distance=d)
