from dataclasses import dataclass
from .obstacles import Obstacle
@dataclass
class World:
    size_x: int
    size_y: int
    obstacles: list[Obstacle]