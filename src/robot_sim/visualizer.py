import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
from numpy.typing import NDArray
from .observation import Observation
from .robot import Robot
from .world import World


class Visualizer:
    def __init__(self, world: World, trail_interval: float = 0.1):
        self.obstacles = world.obstacles
        self.size = [world.size_x, world.size_y]
        # Sim-time seconds between trail points
        self.trail_interval = trail_interval
        self._time_since_trail = trail_interval  # drop a point on the first frame

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, world.size_x)
        self.ax.set_ylim(0, world.size_y)
        for obstacle in self.obstacles:
            self.ax.add_patch(PolygonPatch(obstacle.vertices, closed=True, facecolor="tab:red", alpha=0.4, edgecolor="tab:red"))
        self.trail_x: list[float] = []
        self.trail_y: list[float] = []
        (self.trail_line,) = self.ax.plot([], [], ".", color="tab:gray", markersize=4)
        (self.robot_marker,) = self.ax.plot([], [], "o", color="tab:blue", markersize=10)
        self.range_rays = LineCollection([], colors="tab:cyan", linewidths=0.8, alpha=0.6)
        self.ax.add_collection(self.range_rays)
        self._legend_labels: set[str] = set()
        self.start_pos: NDArray[np.float64] | None = None

    def add_point(self, position: NDArray[np.float64], color: str, label: str | None = None) -> Line2D:
        if label in self._legend_labels:
            label = None
        (marker,) = self.ax.plot([position[0]], [position[1]], "o", color=color, markersize=7, label=label, zorder=3)
        if label is not None:
            self._legend_labels.add(label)
            self.ax.legend(loc="upper left", fontsize="small")
        return marker

    def move_point(self, marker: Line2D, position: NDArray[np.float64]):
        marker.set_data([position[0]], [position[1]])
    def on_click(self, event):
        if self.start_pos is None:
            self.start_pos = np.array([event.xdata, event.ydata], dtype=float)

    def update(self, robot: Robot, dT: float, observation: Observation | None = None):
        x, y = robot.position[0], robot.position[1]
        self.robot_marker.set_data([x], [y])

        # One ray per range measurement, from the robot out to the measured distance.
        # Out-of-range rays are drawn red at max range.
        rays, colors = [], []
        if observation is not None and observation.radial_array is not None:
            radial = observation.radial_array
            for m in radial.measurements:
                in_range = np.isfinite(m.distance)
                if not in_range and not np.isfinite(radial.max_range):
                    continue
                d = m.distance if in_range else radial.max_range
                end = (x + d * np.cos(m.theta), y + d * np.sin(m.theta))
                rays.append([(x, y), end])
                colors.append("tab:cyan" if in_range else "tab:red")
        self.range_rays.set_segments(rays)
        self.range_rays.set_colors(colors)

        self._time_since_trail += dT
        if self._time_since_trail >= self.trail_interval:
            self._time_since_trail = 0.0
            self.trail_x.append(x)
            self.trail_y.append(y)
            self.trail_line.set_data(self.trail_x, self.trail_y)

        self.fig.canvas.draw_idle()
        plt.pause(dT)
