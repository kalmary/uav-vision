from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from numbers import Real
from typing import Optional, Union

import numpy as np

from .depth import DepthResult
from .detection import DetectionResult
from .segmentation import SegmentationResult


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
class ProcessingDiagnostics:
    capture_duration: Optional[float]
    inference_duration: float
    processing_duration: Optional[float]
    raw_count: int
    retained_count: int

    def __post_init__(self) -> None:
        for name, value, optional in (
            ("capture duration", self.capture_duration, True),
            ("inference duration", self.inference_duration, False),
            ("processing duration", self.processing_duration, True),
        ):
            if value is None and optional:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(name + " must be a non-negative finite number")
        for name, value in (
            ("raw count", self.raw_count),
            ("retained count", self.retained_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + " must be a non-negative integer")
        if self.retained_count > self.raw_count:
            raise ValueError("Retained result count must not exceed raw result count")


Result = Union[DetectionResult, SegmentationResult, DepthResult]


@dataclass(frozen=True)
class ProcessedFrame:
    frame: Frame
    result: Result
    diagnostics: ProcessingDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise ValueError("Processed frame must contain a frame")
        if not isinstance(
            self.result, (DetectionResult, SegmentationResult, DepthResult)
        ):
            raise ValueError("Processed frame must contain exactly one result")
        if not isinstance(self.diagnostics, ProcessingDiagnostics):
            raise ValueError("Processed frame must contain diagnostics")
        if isinstance(self.result, SegmentationResult):
            result_height, result_width = self.result.class_map.shape
        elif isinstance(self.result, DepthResult):
            result_height, result_width = self.result.depth_map.shape
        else:
            return
        if result_width != self.frame.width or result_height != self.frame.height:
            raise ValueError("Processed result dimensions must match the frame")

    @property
    def detections(self) -> tuple:
        if not isinstance(self.result, DetectionResult):
            raise AttributeError("Only detection results have detections")
        return self.result.detections
