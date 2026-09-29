import numpy as np
from numpy.typing import NDArray

from .observation import Observation
from .planner.planner import Planner
from .robot import Robot
from .sensors.sensor import ContactSensor, PositionSensor, RadialRangeArray
from .visualizer import Visualizer
from .world import World


class Simulation:
    def __init__(
            self,
            world: World,
            robot: Robot,
            planner: Planner,
            goal: NDArray[np.float64],
            position_sensor: PositionSensor,
            contact_sensor: ContactSensor,
            radial_sensor: RadialRangeArray | None = None,
            visualizer: Visualizer | None = None,
    ):
        self.world = world
        self.robot = robot
        self.planner = planner
        self.position_sensor = position_sensor
        self.contact_sensor = contact_sensor
        self.goal = goal
        self.radial_sensor = radial_sensor

        self.time = 0.0
        self.visualizer = visualizer if visualizer is not None else Visualizer(self.world)

    def step(self, dT: float):
        self.time += dT
        observation = Observation(
            self.position_sensor.measure(self.robot, self.world),
            self.contact_sensor.measure(self.robot, self.world),
            self.radial_sensor.measure(self.robot, self.world)
        )
        self.planner.update(observation)
        self.robot.update(self.planner, dT)
        self.visualizer.update(self.robot, dT, observation)

    def start(self, dT: float, max_steps: int = 2000):
        for _ in range(max_steps):
            if self.planner.done:
                break
            self.step(dT)
