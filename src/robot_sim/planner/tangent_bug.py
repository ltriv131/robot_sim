from dataclasses import dataclass

from .planner import Planner
import numpy as np
from enum import Enum, auto
from numpy.typing import NDArray
from ..visualizer import Visualizer
from ..observation import Observation
import math
class TangentBugState(Enum):
    MOVE_TO_GOAL = auto()
    FOLLOW_PATH = auto()
    
@dataclass
class ObstacleObject:
    left_bound: float
    right_bound: float
    r_left: float
    r_right: float

class TangentBug(Planner):
    def __init__(self, 
                 goal: NDArray[np.float64],
                 goal_tolerance: float = 0.2,
                 offset_tolerance: float = .2,
                 max_range: float = 4.0,
                 jump_threshold: float = 0.5,
                 visualizer: Visualizer | None = None
                 ):
        
        self.goal = np.asarray(goal, dtype=float)
        self.goal_tolerance = goal_tolerance
        self.offset_tolerance = offset_tolerance
        self.max_range = max_range
        self.jump_threshold = jump_threshold
        
        self.visualizer = visualizer
        self.last_following_distance_heuristic = math.inf
        self.m_value: NDArray[np.float64] | None = None

        self.state = TangentBugState.MOVE_TO_GOAL
        self.direction = np.zeros(2)
    def update(self, observation: Observation):
        if observation.radial_array is None:
            raise Exception("Invalid observation for Tangent Bug")
        radialDists = observation.radial_array.measurements
        
        angles, dists = map(list, zip(*[(r.theta, r.distance) for r in radialDists]))
        pos = observation.position.position
        O = self.get_discontinuities(np.array(angles), np.array(dists))
        object_obstacles = self.get_obstacles(O, angles, dists)
        if len(O) == 0:
            self.direction=(self.goal - pos)/(self.dist(self.goal, pos))
        else:
            match self.state:
                case TangentBugState.MOVE_TO_GOAL:
                    self.move_to_goal(object_obstacles, pos, dists, angles)
                    return
                case TangentBugState.FOLLOW_PATH:
                    self.follow_path()
                    return
                case _:
                    return
    def get_obstacles(self, O, angles, dists):
        objects = []
        for i in range(len(O)):
            left = O[i]
            right = O[(i + 1) % len(O)]
            objects.append(ObstacleObject(left_bound=left, right_bound=right, r_left = dists[(np.where(np.asarray(angles) == left)[0][0]) % len(dists)], r_right = dists[(np.where(np.asarray(angles) == right)[0][0]) % len(dists)]))
        objects_obstacles = []
        for o in objects:
            if dists[(np.where(np.asarray(angles) == o.left_bound)[0][0] + 1) % len(dists)] < math.inf:
                objects_obstacles.append(o)
        return objects_obstacles
    def distance_heuristic(self, x, o):
        return self.dist(x, o)+self.dist(o, self.goal)
    
    def dist(self, a, b):
        return np.linalg.norm(b-a)
    def angle_difference(self, alpha, beta):
        # Compute the difference in radians wrapped within [-pi, pi)
        return (alpha - beta + math.pi) % (2 * math.pi) - math.pi
    def move_to_goal(self, O, pos, dists, angles):
        if len(O) == 0:
            self.direction = (self.goal - pos) / self.dist(self.goal, pos)
        
        else:
            
            angle_to_goal = math.atan2(self.goal[1] - pos[1], self.goal[0] - pos[0])
            blocking_o = None
            for o in O:
                if self.angle_difference(angle_to_goal, o.left_bound) > 0 and self.angle_difference(angle_to_goal, o.right_bound) < 0:
                    blocking_o = o
                    break
            if blocking_o is not None:
                print(blocking_o)
                blocking_o_left_cartesian = np.array([blocking_o.r_left*math.cos(blocking_o.left_bound), blocking_o.r_left*math.sin(blocking_o.left_bound)])
                blocking_o_right_cartesian = np.array([blocking_o.r_right*math.cos(blocking_o.right_bound), blocking_o.r_right*math.sin(blocking_o.right_bound)])
                followed_o = blocking_o_left_cartesian if self.distance_heuristic(pos, blocking_o_left_cartesian) < self.distance_heuristic(pos, blocking_o_right_cartesian) else blocking_o_right_cartesian
                self.direction = (followed_o - pos) / self.dist(followed_o, pos)  
                if self.distance_heuristic(pos, followed_o) <= self.last_following_distance_heuristic:
                    
                    self.last_following_distance_heuristic = self.distance_heuristic(pos, followed_o)
                else:
                    self.state = TangentBugState.FOLLOW_PATH
                    self.m_value = min([self.distance_heuristic(pos, self.radial_to_cart(dists[i], angles[i])) for i in range(len(dists))])

            else:
                self.direction = (self.goal - pos) / self.dist(self.goal, pos)
    def radial_to_cart(self, r: float, angle: float) -> np.ndarray:
        return np.array([r * math.cos(angle), r * math.sin(angle)])
    def get_discontinuities(self, angles, dists):
        n = len(dists)
        in_range = dists < self.max_range
        d = np.where(in_range, dists, self.max_range)
        # Compare each ray i with the next one (wrapping around the sweep), and pick the nearer ray of the pair
        i = np.arange(n)
        j = (i + 1) % n
        near_is_i = d[i] <= d[j]
        near = np.where(near_is_i, i, j)
        far = np.where(near_is_i, j, i)
        # Extend the near side's range trend by one ray to predict the far ray. A wall seen at a shallow angle keeps
        # to its trend, and a corner only bends it, so only a real gap between surfaces misses the prediction.
        # If the ray behind the near one is out of range or across another jump there is no trend to extend,
        # so fall back to the near range itself.
        behind = np.where(near_is_i, (i - 1) % n, (j + 1) % n)
        has_trend = in_range[behind] & (np.abs(d[near] - d[behind]) <= self.jump_threshold)
        predicted = np.where(has_trend, 2 * d[near] - d[behind], d[near])
        # An obstacle edge falling out of range is always a jump
        edge = in_range[i] != in_range[j]
        step = in_range[i] & in_range[j] & (np.abs(d[far] - predicted) > self.jump_threshold)
        # Report only the near side of each jump: the endpoint of the obstacle that is actually in view
        return angles[np.unique(near[edge | step])]

    
            
            
    def follow_path(self):
        # TODO: boundary following; keep the current heading until it's implemented
        pass

    def get_direction(self):
        return self.direction