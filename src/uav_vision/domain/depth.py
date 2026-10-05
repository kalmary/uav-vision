from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class DepthResult:
    depth_map: np.ndarray
    unit: str
    scale: float
    minimum: float = field(init=False)
    maximum: float = field(init=False)
    mean: float = field(init=False)
    standard_deviation: float = field(init=False)

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
        depth_map = np.frombuffer(depth_map.tobytes(), dtype=depth_map.dtype).reshape(
            depth_map.shape
        )
        object.__setattr__(self, "depth_map", depth_map)

        values = depth_map.astype(np.float64, copy=False)
        minimum = float(np.minimum.reduce(values.ravel(), dtype=np.float64))
        maximum = float(np.maximum.reduce(values.ravel(), dtype=np.float64))
        magnitude = float(np.maximum.reduce(np.abs(values).ravel(), dtype=np.float64))
        if magnitude == 0.0:
            mean = 0.0
            standard_deviation = 0.0
        else:
            normalized = values / magnitude
            mean = float(np.mean(normalized, dtype=np.float64) * magnitude)
            standard_deviation = float(np.std(normalized, dtype=np.float64) * magnitude)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", standard_deviation)
