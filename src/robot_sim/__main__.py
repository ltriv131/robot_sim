"""Scratch entry point. Run from the project root with: python -m robot_sim"""
import numpy as np

from robot_sim.robot import Robot
from robot_sim.world import World
from robot_sim.obstacles import PolygonObstacle
from robot_sim.simulation import Simulation
from robot_sim.planner.planner import BugPlanner
from robot_sim.sensors.sensor import ContactSensor, PositionSensor, DistanceSensor, RadialRangeArray, RadialDistanceSensor
from robot_sim.visualizer import Visualizer


def star(cx: float, cy: float, r_outer: float, r_inner: float, points: int) -> np.ndarray:
    """Vertices of a star polygon, alternating outer tips and inner notches."""
    angles = np.pi / 2 + np.arange(2 * points) * np.pi / points
    radii = np.where(np.arange(2 * points) % 2 == 0, r_outer, r_inner)
    return np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])


def main():
    goal = np.array([16.5, 17.0])
    obstacles = [
        PolygonObstacle(np.array([[8, 8], [3, 8], [3, 7], [7, 7], [7, 3], [8, 3]], float)),
        PolygonObstacle(star(11.5, 11.5, r_outer=1.8, r_inner=0.8, points=5)),
        PolygonObstacle(np.array([[14, 14], [19, 14], [19, 19], [18, 19], [18, 15], [15, 15], [15, 19], [14, 19]], float)),
        PolygonObstacle(np.array([[13, 3], [17, 2], [18, 6], [16, 5], [14, 7]], float)),
        PolygonObstacle(np.array([[2, 12], [7, 16], [6.4, 16.8], [1.4, 12.8]], float)),
    ]
    world = World(size_x=20, size_y=20, obstacles=obstacles)
    visualizer = Visualizer(world)
    robot = Robot(position=np.array([1.0, 1.0]), orientation=0.0, max_linear_speed=8.0)
    planner = BugPlanner(goal=goal, visualizer=visualizer)
    simulation = Simulation(world, robot=robot, planner=planner, position_sensor=PositionSensor(), contact_sensor=ContactSensor(.2), radial_sensor=RadialDistanceSensor(100), goal=goal, visualizer=visualizer)
    simulation.start(.01)


if __name__ == "__main__":
    main()
