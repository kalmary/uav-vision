from time import perf_counter
from typing import Callable

from uav_vision.config.settings import DetectionFilterSettings
from uav_vision.domain import (
    DetectionResult,
    Frame,
    ProcessedFrame,
    ProcessingDiagnostics,
)
from uav_vision.inference import Detector


class DetectionProcessor:
    def __init__(
        self,
        detector: Detector,
        filters: DetectionFilterSettings = DetectionFilterSettings(),
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._detector = detector
        self._filters = filters
        self._clock = clock

    def process(self, frame: Frame) -> ProcessedFrame:
        started_at = self._clock()
        detections = self._detector.detect(frame)
        inference_duration = self._clock() - started_at
        raw_result = DetectionResult(detections)
        filtered = tuple(
            detection
            for detection in raw_result.detections
            if (
                not self._filters.selected_classes
                or detection.class_id in self._filters.selected_classes
            )
            and detection.confidence >= self._filters.minimum_confidence
        )
        ordered = tuple(
            sorted(filtered, key=lambda detection: detection.confidence, reverse=True)
        )
        if self._filters.top_k is not None:
            ordered = ordered[: self._filters.top_k]
        result = DetectionResult(ordered)
        count = len(raw_result.detections)
        return ProcessedFrame(
            frame=frame,
            result=result,
            diagnostics=ProcessingDiagnostics(
                capture_duration=None,
                inference_duration=inference_duration,
                processing_duration=None,
                raw_count=count,
                retained_count=len(result.detections),
            ),
        )
