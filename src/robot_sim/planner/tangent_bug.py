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
    GOAL_REACHED = auto()
    
@dataclass
class ObstacleObject:
    left_bound: float
    right_bound: float
    r_left: float
    r_right: float

class FollowDirection(Enum):
    CLOCKWISE = auto()
    COUNTERCLOCKWISE = auto()


class TangentBug(Planner):
    def __init__(self, 
                 goal: NDArray[np.float64],
                 goal_tolerance: float = 0.2,
                 offset_tolerance: float = .2,
                 max_range: float = 4.0,
                 jump_threshold: float = 0.5,
                 visualizer: Visualizer | None = None
                 ):
        self.has_hit_wall = False
        self.goal = np.asarray(goal, dtype=float)
        self.goal_tolerance = goal_tolerance
        self.offset_tolerance = offset_tolerance
        self.max_range = max_range
        self.jump_threshold = jump_threshold
        self.follow_distance = 0.7
        self.follow_tolerance = 0.2
        self.d_follow_tolerance = 0.2
        self.current_following_obstacle: ObstacleObject | None = None
        
        self.visualizer = visualizer
        # Markers for the blocking obstacle's left and right Oi in move to goal, created on first use
        self.oi_markers: dict = {}
        self.last_following_distance_heuristic = math.inf
        # d_followed: closest distance to the goal sensed while following an obstacle
        self.m_value: float = math.inf

        self.state = TangentBugState.MOVE_TO_GOAL
        self.follow_path_direction = FollowDirection.CLOCKWISE
        self.direction = np.zeros(2)
    def update(self, observation: Observation):
        if self.dist(observation.position.position, self.goal) < self.goal_tolerance:
                    self.state = TangentBugState.GOAL_REACHED
                    self.direction = np.zeros(2)
                    return
        if observation.radial_array is None:
            raise Exception("Invalid observation for Tangent Bug")
        radialDists = observation.radial_array.measurements
        
        angles, dists = map(list, zip(*[(r.theta, r.distance) for r in radialDists]))
        pos = observation.position.position
        O = self.get_discontinuities(np.array(angles), np.array(dists))
        object_obstacles = self.get_obstacles(O, angles, dists)
        if len(O) == 0:
            self.direction=(self.goal - pos)/(self.dist(self.goal, pos))
            self.show_oi(None)
        else:
            match self.state:
                case TangentBugState.MOVE_TO_GOAL:
                    self.move_to_goal(object_obstacles, pos, dists, angles)
                case TangentBugState.FOLLOW_PATH:
                    self.follow_path(pos, dists, angles, self.current_following_obstacle)
        print(f"{self.state.name:12s} {self.follow_path_direction.name:16s} dir={np.round(self.direction, 2)}")
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
            
            blocking_o = next((o for o in O if self.is_blocking(o, pos, dists, angles)), None)
            if blocking_o is not None:
                
                blocking_o_left_cartesian = self.radial_to_cart(blocking_o.r_left, blocking_o.left_bound, pos)
                blocking_o_right_cartesian = self.radial_to_cart(blocking_o.r_right, blocking_o.right_bound, pos)
                # Aim past each endpoint rather than at it, so the robot clears the corner instead of grazing it
                left_aim = self.offset_oi(blocking_o_left_cartesian, pos, FollowDirection.COUNTERCLOCKWISE)
                right_aim = self.offset_oi(blocking_o_right_cartesian, pos, FollowDirection.CLOCKWISE)
                if self.distance_heuristic(pos, left_aim) < self.distance_heuristic(pos, right_aim):
                    # Passing the lower-angle endpoint keeps the obstacle on the robot's left: a counterclockwise orbit
                    followed_o = left_aim
                    side = FollowDirection.COUNTERCLOCKWISE
                else:
                    followed_o = right_aim
                    side = FollowDirection.CLOCKWISE
                if self.distance_heuristic(pos, followed_o) <= self.last_following_distance_heuristic:
                    self.follow_path_direction = side
                    self.direction = (followed_o - pos) / self.dist(followed_o, pos)
                    self.show_oi(left_aim, right_aim, side)
                    self.last_following_distance_heuristic = self.distance_heuristic(pos, followed_o)
                else:
                    # Keep follow_path_direction from the last decreasing step, so following continues the way
                    # the robot was already going rather than whichever side scores best at the switch
                    self.current_following_obstacle = blocking_o
                    self.has_hit_wall = False
                    self.state = TangentBugState.FOLLOW_PATH
                    self.show_oi(None)
                    # Sample the m point only from rays that hit the blocking obstacle
                    blocking_points = [
                        self.radial_to_cart(dists[i], angles[i], pos)
                        for i in range(len(dists))
                        if dists[i] < math.inf
                        and self.angle_in_range(angles[i], blocking_o.left_bound, blocking_o.right_bound)
                    ]
                    m_point = min(blocking_points, key=lambda p: self.dist(p, self.goal))
                    self.m_value = self.dist(m_point, self.goal)
                    if self.visualizer is not None:
                        self.visualizer.add_point(m_point, "tab:purple", label="m point")

            else:
                self.last_following_distance_heuristic = math.inf
                self.direction = (self.goal - pos) / self.dist(self.goal, pos)
                self.show_oi(None)
    def is_blocking(self, o: ObstacleObject, pos, dists, angles) -> bool:
        # The obstacle blocks only if it is hit before the goal along the bearing to the goal. Being in the same
        # direction is not enough: a wall behind the goal is in range but never in the way.
        angle_to_goal = math.atan2(self.goal[1] - pos[1], self.goal[0] - pos[0])
        if not self.angle_in_range(angle_to_goal, o.left_bound, o.right_bound):
            return False
        # The two rays either side of the bearing to the goal. Both hitting o is what places the goal direction
        # on its surface, so check each that does against the distance to the goal.
        diffs = np.array([self.angle_difference(a, angle_to_goal) for a in angles])
        below = int(np.argmax(np.where(diffs <= 0, diffs, -np.inf)))
        above = int(np.argmin(np.where(diffs > 0, diffs, np.inf)))
        dist_to_goal = self.dist(pos, self.goal)
        return any(
            dists[i] < dist_to_goal and self.angle_in_range(angles[i], o.left_bound, o.right_bound)
            for i in (below, above)
        )
    def show_oi(self, left: NDArray[np.float64] | None, right: NDArray[np.float64] | None = None,
                followed: FollowDirection | None = None):
        # Draw the left and right Oi of the blocking obstacle, the followed one green, or hide both when there are none
        if self.visualizer is None:
            return
        if left is None or right is None:
            for marker in self.oi_markers.values():
                marker.set_visible(False)
            return
        points = {FollowDirection.COUNTERCLOCKWISE: left, FollowDirection.CLOCKWISE: right}
        for side, point in points.items():
            color = "tab:green" if side == followed else "tab:orange"
            if side not in self.oi_markers:
                self.oi_markers[side] = self.visualizer.add_point(point, color)
            else:
                self.visualizer.move_point(self.oi_markers[side], point)
                self.oi_markers[side].set_visible(True)
            self.oi_markers[side].set_color(color)
    def radial_to_cart(self, r: float, angle: float, origin: NDArray[np.float64]) -> np.ndarray:
        # Sensor rays are cast from the robot at world-frame angles, so offset from the robot's position
        return origin + np.array([r * math.cos(angle), r * math.sin(angle)])
    def offset_oi(self, oi: NDArray[np.float64], pos: NDArray[np.float64], side: FollowDirection) -> np.ndarray:
        # Push the endpoint sideways off the obstacle by follow_distance: clockwise of the ray for the
        # lower-angle endpoint, counterclockwise for the higher-angle one, since the obstacle lies between them
        u = (oi - pos) / self.dist(oi, pos)
        if side == FollowDirection.COUNTERCLOCKWISE:
            perp = np.array([u[1], -u[0]])
        else:
            perp = np.array([-u[1], u[0]])
        return oi + self.follow_distance * perp
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

    
            
            
    def angle_in_range(self, angle, start, end):
        # True if angle lies on the counterclockwise sweep from start to end, inclusive of both bounds
        return (angle - start) % (2 * math.pi) <= (end - start) % (2 * math.pi)

    def follow_path(self, pos, dists, angles, follows_obstacle):
        if not self.has_hit_wall:
            # Only consider rays that hit the obstacle being followed
            obstacle_points = [
                self.radial_to_cart(dists[i], angles[i], pos)
                for i in range(len(dists))
                if dists[i] < math.inf
                and self.angle_in_range(angles[i], follows_obstacle.left_bound, follows_obstacle.right_bound)
            ]
            if len(obstacle_points) == 0:
                # Lost sight of the followed obstacle, so head for the goal again
                self.state = TangentBugState.MOVE_TO_GOAL
                return
            closest_point = min(obstacle_points, key=lambda p: self.dist(p, pos))
            if self.dist(pos, closest_point) < self.follow_distance + self.follow_tolerance:
                self.has_hit_wall = True
            else:
                # Head toward the followed obstacle until within follow distance
                self.direction = (closest_point - pos) / self.dist(closest_point, pos)
                return
        viewed_points = [self.radial_to_cart(dists[i], angles[i], pos)
                for i in range(len(dists))
                if dists[i] < math.inf
            ]
        closest_point = min(viewed_points, key=lambda p: self.dist(p, pos))
        # d_reach: distance to the goal from the closest point to it within line of sight
        d_reach = min(self.dist(p, self.goal) for p in viewed_points)
        # Leave once a visible point is closer to the goal than d_followed, which stays fixed while following
        if d_reach < self.m_value - self.d_follow_tolerance:
            self.state = TangentBugState.MOVE_TO_GOAL
            self.last_following_distance_heuristic = math.inf
            if self.visualizer is not None:
                self.visualizer.add_point(pos, "tab:olive", label="leave point")
            return
        
        norm_vector = (pos - closest_point) / self.dist(closest_point, pos)
        transform_matrix = np.array([[1, 0], [0, 1]])
        if abs(self.dist(closest_point, pos) - self.follow_distance) > self.follow_tolerance:
            self.direction = norm_vector * -1.0*np.sign(self.dist(closest_point, pos) - self.follow_distance)
            return
        # Rotating the outward normal +90 degrees orbits the obstacle counterclockwise, -90 clockwise
        if self.follow_path_direction == FollowDirection.COUNTERCLOCKWISE:
            transform_matrix = np.array([[0, -1], [1, 0]])
        else:
            transform_matrix = np.array([[0, 1], [-1, 0]])
        tangent_vector = transform_matrix @ norm_vector
        self.direction = tangent_vector
        
    def get_direction(self):
        return self.direction