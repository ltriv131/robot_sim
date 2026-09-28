from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import math
from .planner.planner import Planner

@dataclass
class Robot:
    position: NDArray[np.float64]
    orientation: float
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0

    def move2d(self, velx: float, vely: float, dT: float):
        vels = self.clampVel(velx, vely)
        self.position += vels * dT

    def clampVel(self, *args):
        return np.array([max(-1.0 * self.max_linear_speed, min(self.max_linear_speed, arg)) for arg in args])

    def update(self, planner: Planner, dT: float):
        direction = planner.get_direction()
        norm = np.linalg.norm(direction)
        if norm == 0:
            return
        vel = direction / norm * self.max_linear_speed
        self.move2d(*vel, dT)
        

    
