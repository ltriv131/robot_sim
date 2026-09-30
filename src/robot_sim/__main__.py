"""Scratch entry point. Run from the project root with: python -m robot_sim"""
import numpy as np

from robot_sim.robot import Robot
from robot_sim.world import World
from robot_sim.obstacles import PolygonObstacle
from robot_sim.simulation import Simulation
from robot_sim.planner.planner import BugPlanner
from robot_sim.planner.tangent_bug import TangentBug
from robot_sim.sensors.sensor import ContactSensor, PositionSensor, DistanceSensor, RadialRangeArray, RadialDistanceSensor
from robot_sim.visualizer import Visualizer


def star(cx: float, cy: float, r_outer: float, r_inner: float, points: int) -> np.ndarray:
    """Vertices of a star polygon, alternating outer tips and inner notches."""
    angles = np.pi / 2 + np.arange(2 * points) * np.pi / points
    radii = np.where(np.arange(2 * points) % 2 == 0, r_outer, r_inner)
    return np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])


def regular_polygon(cx: float, cy: float, r: float, sides: int, rotation: float = 0.0) -> np.ndarray:
    """Vertices of a regular polygon with circumradius r."""
    angles = rotation + np.arange(sides) * 2 * np.pi / sides
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def trap_obstacles() -> list[PolygonObstacle]:
    """Concave and wide shapes that make the heuristic rise, forcing obstacle following."""
    return [
        PolygonObstacle(np.array([[8, 8], [3, 8], [3, 7], [7, 7], [7, 3], [8, 3]], float)),
        PolygonObstacle(star(11.5, 11.5, r_outer=1.8, r_inner=0.8, points=5)),
        PolygonObstacle(np.array([[14, 14], [19, 14], [19, 19], [18, 19], [18, 15], [15, 15], [15, 19], [14, 19]], float)),
        PolygonObstacle(np.array([[13, 3], [17, 2], [18, 6], [16, 5], [14, 7]], float)),
        PolygonObstacle(np.array([[2, 12], [7, 16], [6.4, 16.8], [1.4, 12.8]], float)),
    ]


def glide_obstacles() -> list[PolygonObstacle]:
    """Small convex shapes on the start-goal line. Heading for an edge keeps the heuristic falling,
    so the robot should glide around them without switching to obstacle following."""
    return [
        # Octagon straddling the start-goal line
        PolygonObstacle(regular_polygon(5.0, 5.3, r=1.0, sides=8, rotation=np.pi / 8)),
        # Diamond pointing at the robot, so its faces deflect it sideways
        PolygonObstacle(regular_polygon(9.5, 10.0, r=1.2, sides=4, rotation=np.deg2rad(46.8))),
        # Short thin wall across the path, only a small detour past either end
        PolygonObstacle(np.array([[12.12, 14.47], [13.67, 12.91], [13.89, 13.13], [12.33, 14.69]], float)),
        # Off-path scenery
        PolygonObstacle(regular_polygon(15.0, 6.0, r=1.5, sides=6)),
        PolygonObstacle(regular_polygon(5.0, 14.0, r=1.5, sides=3, rotation=np.pi / 2)),
    ]


def main():
    goal = np.array([16.5, 17.0])
    obstacles = glide_obstacles()
    world = World(size_x=20, size_y=20, obstacles=glide_obstacles())
    visualizer = Visualizer(world)
    robot = Robot(position=np.array([1.0, 1.0]), orientation=0.0, max_linear_speed=8.0)
    planner = BugPlanner(goal=goal, visualizer=visualizer)
    planner_tangent = TangentBug(goal=goal, visualizer=visualizer)
    simulation = Simulation(world, robot=robot, planner=planner_tangent, position_sensor=PositionSensor(), contact_sensor=ContactSensor(.2), radial_sensor=RadialDistanceSensor(100, max_range = 4), goal=goal, visualizer=visualizer)
    simulation.start(.01)


if __name__ == "__main__":
    main()
