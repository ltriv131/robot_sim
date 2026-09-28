from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from robot_sim.observation import Observation

if TYPE_CHECKING:
    # Type-only: visualizer imports robot, which imports this module
    from robot_sim.visualizer import Visualizer


class Planner:
    def update(self, observation: Observation) -> None:
        raise NotImplementedError

    def get_direction(self) -> NDArray[np.float64]:
        """Unit heading for the robot, or the zero vector to stop."""
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return False


class BugState(Enum):
    GO_TO_GOAL = auto()
    CIRCUMNAVIGATE = auto()      # first full lap, recording the closest point
    RETURN_TO_CLOSEST = auto()   # keep following the wall back to that point
    LEAVE = auto()               # head for the goal, ignoring contact until clear
    REACHED = auto()


class BugPlanner(Planner):
    """Bug1: on hitting an obstacle, walk all the way around it, then leave
    from the boundary point closest to the goal."""

    def __init__(
            self,
            goal: NDArray[np.float64],
            goal_tolerance: float = 0.2,
            point_tolerance: float = 0.4,
            recovery_turn: float = np.radians(45),
            visualizer: Visualizer | None = None,
    ):
        self.goal = np.asarray(goal, dtype=float)
        self.goal_tolerance = goal_tolerance
        self.point_tolerance = point_tolerance
        self.recovery_turn = recovery_turn
        self.visualizer = visualizer
        self._closest_marker = None

        self.state = BugState.GO_TO_GOAL
        self.direction = np.zeros(2)
        self.hit_point = np.zeros(2)
        self.closest_point = np.zeros(2)
        self.left_hit_point = False

    def set_goal(self, goal: NDArray[np.float64]) -> None:
        self.goal = np.asarray(goal, dtype=float)
        self.state = BugState.GO_TO_GOAL

    @property
    def done(self) -> bool:
        return self.state == BugState.REACHED

    def get_direction(self) -> NDArray[np.float64]:
        return self.direction

    def update(self, observation: Observation) -> None:
        # Copy: the sensor hands back the robot's own array, which moves in place
        position = np.array(observation.position.position, dtype=float)
        contact = observation.contact
        touching = contact is not None and contact.contact

        if self._dist_to_goal(position) < self.goal_tolerance:
            self.state = BugState.REACHED

        if self.state == BugState.REACHED:
            self.direction = np.zeros(2)

        elif self.state == BugState.GO_TO_GOAL:
            if touching:
                self._start_circumnavigating(position)
                self._follow_wall(contact)
            else:
                self._head_to_goal(position)

        elif self.state == BugState.CIRCUMNAVIGATE:
            if self._dist_to_goal(position) < self._dist_to_goal(self.closest_point):
                self.closest_point = position
                if self.visualizer is not None:
                    self.visualizer.move_point(self._closest_marker, position)

            dist_to_hit = np.linalg.norm(position - self.hit_point)
            if not self.left_hit_point and dist_to_hit > 2 * self.point_tolerance:
                self.left_hit_point = True
            elif self.left_hit_point and dist_to_hit < self.point_tolerance:
                self.state = BugState.RETURN_TO_CLOSEST
            self._follow_wall(contact)

        elif self.state == BugState.RETURN_TO_CLOSEST:
            if np.linalg.norm(position - self.closest_point) < self.point_tolerance:
                self.state = BugState.LEAVE
                self._head_to_goal(position)
            else:
                self._follow_wall(contact)

        elif self.state == BugState.LEAVE:
            if not touching:
                self.state = BugState.GO_TO_GOAL
            self._head_to_goal(position)

    def _start_circumnavigating(self, position: NDArray[np.float64]) -> None:
        self.state = BugState.CIRCUMNAVIGATE
        self.hit_point = position
        self.closest_point = position
        self.left_hit_point = False
        if self.visualizer is not None:
            # One hit dot per obstacle; the closest dot moves as the lap finds better points
            self.visualizer.add_point(position, "tab:orange", label="hit point")
            self._closest_marker = self.visualizer.add_point(position, "tab:green", label="closest point")

    def _follow_wall(self, contact) -> None:
        if contact is not None and contact.contact:
            self.direction = contact.tangent
        else:
           
            c, s = np.cos(self.recovery_turn), np.sin(self.recovery_turn)
            x, y = self.direction
            self.direction = np.array([c * x + s * y, -s * x + c * y])

    def _head_to_goal(self, position: NDArray[np.float64]) -> None:
        to_goal = self.goal - position
        self.direction = to_goal / np.linalg.norm(to_goal)

    def _dist_to_goal(self, position: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(self.goal - position))
