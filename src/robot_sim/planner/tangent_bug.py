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

class FollowDirection(Enum):
    CLOCKWISE = auto()
    COUNTERCLOCKWISE = auto()
    GOAL_REACHED = auto()

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
        self.follow_distance = 1.0
        self.follow_tolerance = 0.2
        
        self.visualizer = visualizer
        self.last_following_distance_heuristic = math.inf
        self.m_value: NDArray[np.float64] | None = None

        self.state = TangentBugState.MOVE_TO_GOAL
        self.follow_path_direction = FollowDirection.CLOCKWISE
        self.direction = np.zeros(2)
    def update(self, observation: Observation):
        print(self.state)
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
                    self.follow_path(pos, dists, angles)
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
                blocking_o_left_cartesian = self.radial_to_cart(blocking_o.r_left, blocking_o.left_bound, pos)
                blocking_o_right_cartesian = self.radial_to_cart(blocking_o.r_right, blocking_o.right_bound, pos)
                if self.distance_heuristic(pos, blocking_o_left_cartesian) < self.distance_heuristic(pos, blocking_o_right_cartesian):
                    followed_o = blocking_o_left_cartesian
                    self.follow_path_direction = FollowDirection.CLOCKWISE
                else:
                    followed_o = blocking_o_right_cartesian
                    self.follow_path_direction = FollowDirection.COUNTERCLOCKWISE
                self.direction = (followed_o - pos) / self.dist(followed_o, pos)  
                if self.distance_heuristic(pos, followed_o) <= self.last_following_distance_heuristic:
                    
                    self.last_following_distance_heuristic = self.distance_heuristic(pos, followed_o)
                else:
                    self.state = TangentBugState.FOLLOW_PATH
                    sensed_points = [self.radial_to_cart(dists[i], angles[i], pos) for i in range(len(dists)) if dists[i] < math.inf]
                    m_point = min(sensed_points, key=lambda p: self.distance_heuristic(pos, p))
                    self.m_value = self.distance_heuristic(pos, m_point)
                    if self.visualizer is not None:
                        self.visualizer.add_point(m_point, "tab:purple", label="m point")

            else:
                self.last_following_distance_heuristic = math.inf
                self.direction = (self.goal - pos) / self.dist(self.goal, pos)
    def radial_to_cart(self, r: float, angle: float, origin: NDArray[np.float64]) -> np.ndarray:
        # Sensor rays are cast from the robot at world-frame angles, so offset from the robot's position
        return origin + np.array([r * math.cos(angle), r * math.sin(angle)])
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

    
            
            
    def follow_path(self, pos, dists, angles):
        
        closest_point = min([self.radial_to_cart(dists[i], angles[i], pos) for i in range(len(dists)) if dists[i] < math.inf], key=lambda p: self.dist(p, pos))
        if self.distance_heuristic(pos, closest_point) < self.m_value:
            self.state = TangentBugState.MOVE_TO_GOAL
            if self.visualizer is not None:
                self.visualizer.add_point(pos, "tab:olive", label="leave point")
            return
        if self.dist(pos, self.goal) < self.goal_tolerance:
            self.state = TangentBugState.GOAL_REACHED
            return
        norm_vector = (pos - closest_point) / self.dist(closest_point, pos)
        transform_matrix = np.array([[1, 0], [0, 1]])
        if abs(self.dist(closest_point, pos) - self.follow_distance) > self.follow_tolerance:
            self.direction = norm_vector * -1.0*np.sign(self.dist(closest_point, pos) - self.follow_distance)
            return
        if self.follow_path_direction == FollowDirection.COUNTERCLOCKWISE:
            transform_matrix = np.array([[0, 1], [-1, 0]])
        else:
            transform_matrix = np.array([[0, -1], [1, 0]])
        tangent_vector = transform_matrix @ norm_vector
        self.direction = tangent_vector
        
    def get_direction(self):
        return self.direction