"""Scratch entry point. Run from the project root with: python -m robot_sim [scenario]"""
import sys

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from robot_sim.robot import Robot
from robot_sim.world import World
from robot_sim.obstacles import PolygonObstacle
from robot_sim.simulation import Simulation
from robot_sim.planner.planner import BugPlanner
from robot_sim.planner.tangent_bug import TangentBug
from robot_sim.sensors.sensor import ContactSensor, PositionSensor, DistanceSensor, RadialRangeArray, RadialDistanceSensor
from robot_sim.visualizer import Visualizer
import matplotlib.pyplot as plt

def star(cx: float, cy: float, r_outer: float, r_inner: float, points: int) -> np.ndarray:
    """Vertices of a star polygon, alternating outer tips and inner notches."""
    angles = np.pi / 2 + np.arange(2 * points) * np.pi / points
    radii = np.where(np.arange(2 * points) % 2 == 0, r_outer, r_inner)
    return np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])


def regular_polygon(cx: float, cy: float, r: float, sides: int, rotation: float = 0.0) -> np.ndarray:
    """Vertices of a regular polygon with circumradius r."""
    angles = rotation + np.arange(sides) * 2 * np.pi / sides
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def gear(cx: float, cy: float, r_root: float, r_tip: float, teeth: int) -> np.ndarray:
    """Vertices of a gear with trapezoidal teeth."""
    pitch = 2 * np.pi / teeth
    offsets = np.array([0.0, 0.2, 0.45, 0.65]) * pitch
    radii = np.array([r_root, r_tip, r_tip, r_root])
    angles = (np.arange(teeth)[:, None] * pitch + offsets).ravel()
    r = np.tile(radii, teeth)
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def spiral(cx: float, cy: float, r_start: float, r_end: float, turns: float, width: float, samples_per_turn: int = 10) -> np.ndarray:
    """Vertices of a thick Archimedean spiral arm winding outward counterclockwise.
    The pitch (r_end - r_start) / turns must exceed width or the arms overlap."""
    n = int(samples_per_turn * turns) + 1
    theta = np.linspace(0, 2 * np.pi * turns, n)
    r = np.linspace(r_start, r_end, n)
    outer = np.column_stack([cx + (r + width / 2) * np.cos(theta), cy + (r + width / 2) * np.sin(theta)])
    inner = np.column_stack([cx + (r - width / 2) * np.cos(theta), cy + (r - width / 2) * np.sin(theta)])
    return np.vstack([outer, inner[::-1]])


def thick_path(*polylines: list[tuple[float, float]], width: float) -> np.ndarray:
    """Vertices of the union of polylines thickened to the given width, with square ends and mitred joints."""
    shape = unary_union([LineString(p).buffer(width / 2, cap_style="flat", join_style="mitre") for p in polylines])
    return np.array(shape.exterior.coords[:-1])


def crescent(cx: float, cy: float, r: float, bite: float, direction: float) -> np.ndarray:
    """Vertices of a disc with a smaller-offset disc removed, leaving a cup that opens toward direction."""
    d = np.array([np.cos(direction), np.sin(direction)])
    shape = Point(cx, cy).buffer(r, quad_segs=3).difference(Point(cx + bite * d[0], cy + bite * d[1]).buffer(r * 0.9, quad_segs=3))
    return np.array(shape.exterior.coords[:-1])


def cup_obstacles() -> list[PolygonObstacle]:
    """The goal sits inside a C that opens away from the robot, behind a comb and a zigzag wall.
    Every obstacle has pockets that are local minima of the heuristic."""
    return [
        # Comb with teeth pointing at the start, leaving dead-end pockets between them
        PolygonObstacle(thick_path([(3.0, 6.0), (8.5, 6.0)], [(3.5, 6.0), (3.5, 3.0)], [(5.5, 6.0), (5.5, 3.0)], [(7.5, 6.0), (7.5, 3.0)], width=0.5)),
        # Zigzag wall across the start-goal line
        PolygonObstacle(thick_path([(4.0, 11.0), (6.0, 8.5), (8.0, 11.0), (10.0, 8.5), (12.0, 11.0)], width=0.5)),
        # C around the goal, open on the far side
        PolygonObstacle(thick_path([(18.3, 15.0), (13.8, 15.0), (13.8, 19.0), (18.3, 19.0)], width=0.6)),
        # Off-path scenery
        PolygonObstacle(gear(15.5, 6.5, r_root=1.2, r_tip=1.7, teeth=6)),
        PolygonObstacle(star(4.5, 15.5, r_outer=1.8, r_inner=0.7, points=6)),
    ]


def spiral_obstacles() -> list[PolygonObstacle]:
    """The goal is at the centre of a spiral, so the robot must wind in along the arm.
    On the way it meets a star and a crescent cupped toward it."""
    return [
        PolygonObstacle(star(4.5, 4.5, r_outer=1.6, r_inner=0.6, points=4)),
        PolygonObstacle(crescent(9.5, 9.5, r=2.0, bite=1.2, direction=np.deg2rad(225))),
        PolygonObstacle(spiral(16.5, 17.0, r_start=1.0, r_end=2.6, turns=1.25, width=0.4)),
        # Off-path scenery
        PolygonObstacle(regular_polygon(15.0, 6.0, r=1.6, sides=6)),
        PolygonObstacle(regular_polygon(5.0, 15.0, r=1.3, sides=5, rotation=np.pi / 2)),
    ]


def clutter_obstacles() -> list[PolygonObstacle]:
    """A dense field of mixed shapes, including a narrow gap on the start-goal line
    and several concave pieces that can trap a greedy planner."""
    return [
        # Slanted plate across the line right after the start
        PolygonObstacle(thick_path([(2.0, 6.0), (6.0, 3.0)], width=0.6)),
        PolygonObstacle(gear(9.0, 3.5, r_root=1.0, r_tip=1.4, teeth=6)),
        PolygonObstacle(star(3.0, 10.0, r_outer=1.5, r_inner=0.6, points=4)),
        # L-shape whose inner corner faces the start
        PolygonObstacle(thick_path([(6.5, 10.5), (9.5, 10.5), (9.5, 7.0)], width=0.6)),
        # Two plates leaving a ~2 wide gap on the start-goal line
        PolygonObstacle(thick_path([(9.5, 14.8), (11.6, 12.7)], width=0.6)),
        PolygonObstacle(thick_path([(13.0, 11.3), (15.0, 9.3)], width=0.6)),
        PolygonObstacle(crescent(14.5, 4.5, r=1.6, bite=1.0, direction=np.deg2rad(150))),
        PolygonObstacle(regular_polygon(18.0, 10.0, r=1.2, sides=6)),
        PolygonObstacle(regular_polygon(7.0, 17.5, r=1.3, sides=5, rotation=np.pi / 2)),
        # Hook just short of the goal
        PolygonObstacle(thick_path([(13.0, 15.5), (15.0, 15.5), (15.0, 18.5), (14.0, 18.5)], width=0.5)),
        PolygonObstacle(star(18.5, 14.0, r_outer=1.0, r_inner=0.45, points=5)),
        PolygonObstacle(regular_polygon(10.5, 18.0, r=1.1, sides=3, rotation=np.pi / 2)),
    ]


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


SCENARIOS = {
    "glide": glide_obstacles,
    "trap": trap_obstacles,
    "cup": cup_obstacles,
    "spiral": spiral_obstacles,
    "clutter": clutter_obstacles,
}


def main():
    goal = np.array([16.5, 17.0])
    scenario = sys.argv[1] if len(sys.argv) > 1 else "spiral"
    world = World(size_x=20, size_y=20, obstacles=SCENARIOS[scenario]())
    visualizer = Visualizer(world, trail_interval=0.05)
    visualizer.add_point(goal, "gold", label="goal", marker_style="*", markersize=16)
    robot = Robot(position=np.array([1.0, 1.0]), orientation=0.0, max_linear_speed=4.0)
    planner = BugPlanner(goal=goal, visualizer=visualizer)
    planner_tangent = TangentBug(goal=goal, visualizer=visualizer)
    simulation = Simulation(world, robot=robot, planner=planner_tangent, position_sensor=PositionSensor(), contact_sensor=ContactSensor(.2), radial_sensor=RadialDistanceSensor(100, max_range = 4), goal=goal, visualizer=visualizer)
    while visualizer.start_pos is None:
        plt.pause(0.01)
    robot.position = visualizer.start_pos
    simulation.start(.01)


if __name__ == "__main__":
    main()
