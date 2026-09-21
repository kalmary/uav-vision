from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SegmentationClass:
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise ValueError("Segmentation class identifier must be non-negative")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("Segmentation class name must be non-empty")


@dataclass(frozen=True)
class SegmentationResult:
    class_map: np.ndarray
    classes: Tuple[SegmentationClass, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.class_map, np.ndarray)
            or self.class_map.ndim != 2
            or self.class_map.size == 0
            or not np.issubdtype(self.class_map.dtype, np.integer)
            or np.any(self.class_map < 0)
        ):
            raise ValueError(
                "Segmentation class map must be a non-empty 2D integer array"
            )
        if not isinstance(self.classes, tuple) or not all(
            isinstance(value, SegmentationClass) for value in self.classes
        ):
            raise ValueError("Segmentation classes must be a tuple of class metadata")

        class_ids = {value.class_id for value in self.classes}
        if len(class_ids) != len(self.classes):
            raise ValueError("Segmentation class metadata must not repeat identifiers")
        if not set(np.unique(self.class_map)).issubset(class_ids):
            raise ValueError(
                "Segmentation class map contains unknown class identifiers"
            )
        class_map = np.array(self.class_map, copy=True, order="C")
        class_map.setflags(write=False)
        object.__setattr__(self, "class_map", class_map)
