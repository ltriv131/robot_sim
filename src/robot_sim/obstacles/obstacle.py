from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points
import math

class Obstacle(Protocol):
    def within(self, r, pos) -> bool:
        ...

    def tangent(self, pos, standoff: float | None = None) -> NDArray[np.float64]:
        ...


@dataclass
class PolygonObstacle:
    vertices: NDArray
    shape: Polygon = field(init=False)

    def __post_init__(self):
        # Clockwise exterior, used by the on-boundary fallback in tangent()
        self.shape = orient(Polygon(self.vertices), sign=-1.0)

    def within(self, r, pos):
        return self.shape.distance(Point(pos)) < r

    def tangent(self, pos, standoff: float | None = None, gain: float = 2.0):
        """Unit vector along the boundary, going clockwise, at the closest point to pos.

        If standoff is given, the vector is bent toward/away from the wall to hold
        that distance, so following it doesn't drift off convex corners.
        """
        p = np.asarray(pos, dtype=float)
        q = np.array(nearest_points(self.shape.exterior, Point(p))[0].coords[0])
        n = p - q
        dist = np.linalg.norm(n)
        if dist < 1e-9:
            return self._edge_tangent(p)
        n /= dist
        if self.shape.contains(Point(p)):
            # Robot has ended up inside: flip so n is still the outward normal
            # and the standoff term pushes it back out.
            n = -n
            dist = -dist
        # Outward normal rotated -90 degrees
        t = np.array([n[1], -n[0]])
        if standoff is not None:
            t = t - gain * (dist - standoff) * n
            t /= np.linalg.norm(t)
        return t

    def _edge_tangent(self, p, eps: float = 1e-6):
        ring = self.shape.exterior
        d = ring.project(Point(p))
        a = np.array(ring.interpolate(d).coords[0])
        b = np.array(ring.interpolate((d + eps) % ring.length).coords[0])
        t = b - a
        return t / np.linalg.norm(t)
    def intersect(self, pos: NDArray[np.float64], theta: float, max_range: float) -> float:
        """Distance along the ray from pos at angle theta to the nearest edge, or inf if none within max_range."""
        e = max_range * np.array([math.cos(theta), math.sin(theta)])
        closest = math.inf
        for ind, v in enumerate(self.vertices):
            v_1 = self.vertices[(ind + 1) % len(self.vertices)]
            f = np.asarray(v_1, dtype=float) - v

            # Solve v + s*f = pos + t*e for (s, t)
            A = np.array([
                [f[0], -e[0]],
                [f[1], -e[1]]
            ])
            if abs(np.linalg.det(A)) < 1e-12:
                continue  # ray parallel to edge
            b = np.asarray(pos, dtype=float) - v
            s, t = np.linalg.solve(A, b)
            if 0 <= s <= 1 and 0 < t <= 1:
                closest = min(closest, t * max_range)
        return closest
