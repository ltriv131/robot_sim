from __future__ import annotations
from dataclasses import dataclass 
import numpy as np
from numpy.typing import NDArray
from typing import Protocol

@dataclass
class Observation:
    position: PositionMeasurement
    contact: ContactMeasurement | None
    
@dataclass
class BugObservation(Observation):
    pass

class Measurement(Protocol):
    pass

@dataclass
class PositionMeasurement:
    position: NDArray[np.float64]

@dataclass
class ContactMeasurement:
    contact: bool
    tangent: NDArray[np.float64]



