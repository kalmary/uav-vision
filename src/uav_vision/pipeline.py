from dataclasses import replace
from time import perf_counter, sleep
from typing import Callable, Optional, Sequence

from uav_vision.capture.base import FrameSource
from uav_vision.output.base import FrameOutput
from uav_vision.processing.base import FrameProcessor

_MAX_STOP_WAIT = 0.05


def _wait_until(
    deadline: float,
    should_stop: Optional[Callable[[], bool]],
    clock: Callable[[], float],
    wait: Callable[[float], None],
) -> bool:
    remaining = deadline - clock()
    if remaining <= 0:
        return False
    if should_stop is None:
        wait(remaining)
        return False
    if should_stop():
        return True
    while remaining > 0:
        wait(min(remaining, _MAX_STOP_WAIT))
        if should_stop():
            return True
        remaining = deadline - clock()
    return False


def run_pipeline(
    source: FrameSource,
    processor: FrameProcessor,
    outputs: Sequence[FrameOutput],
    should_stop: Optional[Callable[[], bool]] = None,
    fps: int = 0,
    clock: Callable[[], float] = perf_counter,
    wait: Callable[[float], None] = sleep,
) -> None:
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 0:
        raise ValueError("fps must be a non-negative integer")
    previous_frame_started_at = None
    while True:
        if should_stop is not None and should_stop():
            return

        frame_started_at = clock()
        frame = source.read()
        if frame is None:
            return
        capture_duration = clock() - frame_started_at
        frames_per_second = None
        if previous_frame_started_at is not None:
            frame_period = frame_started_at - previous_frame_started_at
            if frame_period > 0:
                frames_per_second = 1.0 / frame_period
        previous_frame_started_at = frame_started_at

        processing_started_at = clock()
        processed = processor.process(frame)
        processing_duration = clock() - processing_started_at
        processed = replace(
            processed,
            diagnostics=replace(
                processed.diagnostics,
                capture_duration=capture_duration,
                processing_duration=processing_duration,
                frames_per_second=frames_per_second,
            ),
        )
        output_requested_stop = False
        for output in outputs:
            if output.write(processed):
                output_requested_stop = True

        if output_requested_stop:
            return
        if fps > 0:
            deadline = frame_started_at + (1.0 / fps)
            if _wait_until(deadline, should_stop, clock, wait):
                return
