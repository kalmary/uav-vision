from dataclasses import replace
from time import perf_counter
from typing import Callable, Optional, Sequence

from uav_vision.capture.base import FrameSource
from uav_vision.output.base import FrameOutput
from uav_vision.processing.base import FrameProcessor


def run_pipeline(
    source: FrameSource,
    processor: FrameProcessor,
    outputs: Sequence[FrameOutput],
    should_stop: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = perf_counter,
) -> None:
    while True:
        if should_stop is not None and should_stop():
            return

        capture_started_at = clock()
        frame = source.read()
        capture_duration = clock() - capture_started_at
        if frame is None:
            return

        processing_started_at = clock()
        processed = processor.process(frame)
        processing_duration = clock() - processing_started_at
        processed = replace(
            processed,
            diagnostics=replace(
                processed.diagnostics,
                capture_duration=capture_duration,
                processing_duration=processing_duration,
            ),
        )
        output_requested_stop = False
        for output in outputs:
            if output.write(processed):
                output_requested_stop = True

        if output_requested_stop:
            return
