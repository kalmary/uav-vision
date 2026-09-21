from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class DepthResult:
    depth_map: np.ndarray
    unit: str
    scale: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.depth_map, np.ndarray)
            or self.depth_map.ndim != 2
            or self.depth_map.size == 0
            or not np.issubdtype(self.depth_map.dtype, np.floating)
            or not np.isfinite(self.depth_map).all()
        ):
            raise ValueError("Depth map must be a non-empty finite 2D float array")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("Depth unit must be non-empty")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, Real)
            or not isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("Depth scale must be a positive finite number")
        depth_map = np.array(self.depth_map, copy=True, order="C")
        depth_map.setflags(write=False)
        object.__setattr__(self, "depth_map", depth_map)
