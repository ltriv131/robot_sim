from .planner import Planner
import numpy as np
from enum import Enum, auto
from numpy.typing import NDArray
from ..visualizer import Visualizer
from ..observation import Observation
import math
from pandas import Series
class TangentBugState(Enum):
    MOVE_TO_GOAL = auto()
    FOLLOW_PATH = auto()

class TangentBug(Planner):
    def __init__(self, 
                 goal: NDArray[np.float64],
                 goal_tolerance: float = 0.2,
                 offset_tolerance: float = .2,
                 visualizer: Visualizer | None = None
                 ):
        
        self.goal = np.asarray(goal, dtype=float)
        self.goal_tolerance = goal_tolerance
        self.offset_tolerance = offset_tolerance
        
        self.visualizer = visualizer
        

        self.state = TangentBugState.MOVE_TO_GOAL
        self.direction = np.zeros(2)
    def update(self, observation: Observation):
        if observation.radial_array is None:
            raise Exception("Invalid observation for Tangent Bug")
        radialDists = observation.radial_array.measurements
        angles, dists = map(list, zip(*[(r.theta, r.distance) for r in radialDists]))
        pos = observation.position.position
        O = self.get_discontinuities(angles, dists) + pos
        if O is np.empty:
            self.direction=(self.goal - pos)/(self.dist(self.goal, pos))
        else:
            match self.state:
                case TangentBugState.MOVE_TO_GOAL:
                    self.move_to_goal()
                    return
                case TangentBugState.FOLLOW_PATH:
                    self.follow_path()
                    return
                case _:
                    return
    def distance_heuristic(self, x, o):
        return self.dist(x, o)+self.dist(o, self.goal)
    
    def dist(self, a, b):
        return np.linalg.norm(b-a)
    
    
        
    def get_direction(self):
        return self.direction