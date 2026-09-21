from time import perf_counter
from typing import Callable

from uav_vision.domain import (
    DetectionResult,
    Frame,
    ProcessedFrame,
    ProcessingDiagnostics,
)
from uav_vision.inference import Detector


class DetectionProcessor:
    def __init__(
        self, detector: Detector, clock: Callable[[], float] = perf_counter
    ) -> None:
        self._detector = detector
        self._clock = clock

    def process(self, frame: Frame) -> ProcessedFrame:
        started_at = self._clock()
        detections = self._detector.detect(frame)
        inference_duration = self._clock() - started_at
        result = DetectionResult(detections)
        count = len(result.detections)
        return ProcessedFrame(
            frame=frame,
            result=result,
            diagnostics=ProcessingDiagnostics(
                capture_duration=None,
                inference_duration=inference_duration,
                processing_duration=None,
                raw_count=count,
                retained_count=count,
            ),
        )
