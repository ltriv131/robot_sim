from typing import Protocol

import numpy as np

from ..observation import ContactMeasurement, Measurement, PositionMeasurement
from ..robot import Robot
from ..world import World


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
