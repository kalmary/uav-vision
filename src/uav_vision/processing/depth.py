from time import perf_counter
from typing import Callable

from uav_vision.config.settings import ProcessingType
from uav_vision.domain import Frame, ProcessedFrame, ProcessingDiagnostics
from uav_vision.inference import DepthEstimator


class DepthProcessor:
    def __init__(
        self,
        estimator: DepthEstimator,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._estimator = estimator
        self._clock = clock

    def process(self, frame: Frame) -> ProcessedFrame:
        started_at = self._clock()
        result = self._estimator.estimate_depth(frame)
        inference_duration = self._clock() - started_at
        count = int(result.depth_map.size)
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
            processing_type=ProcessingType.DEPTH,
        )
