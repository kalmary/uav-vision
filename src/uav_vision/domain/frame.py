from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np

from .detection import Detection


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    sequence: int
    captured_at: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.image, np.ndarray)
            or self.image.ndim != 3
            or self.image.shape[2] != 3
            or self.image.dtype != np.uint8
            or not self.image.flags.c_contiguous
            or self.image.size == 0
        ):
            raise ValueError(
                "Frame image must be a non-empty contiguous BGR uint8 array"
            )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("Frame sequence must be non-negative")
        if (
            not isinstance(self.captured_at, datetime)
            or self.captured_at.tzinfo is None
            or self.captured_at.utcoffset() is None
        ):
            raise ValueError("Frame capture timestamp must include a timezone")

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True)
class ProcessedFrame:
    frame: Frame
    detections: Tuple[Detection, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise ValueError("Processed frame must contain a frame")
        if not isinstance(self.detections, tuple) or not all(
            isinstance(detection, Detection) for detection in self.detections
        ):
            raise ValueError(
                "Processed frame detections must be a tuple of Detection values"
            )
