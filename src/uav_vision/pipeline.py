from typing import Callable, Optional, Sequence

from uav_vision.capture import FrameSource
from uav_vision.output import FrameOutput
from uav_vision.processing import FrameProcessor


def run_pipeline(
    source: FrameSource,
    processor: FrameProcessor,
    outputs: Sequence[FrameOutput],
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    while True:
        if should_stop is not None and should_stop():
            return

        frame = source.read()
        if frame is None:
            return

        processed = processor.process(frame)
        output_requested_stop = False
        for output in outputs:
            if output.write(processed):
                output_requested_stop = True

        if output_requested_stop:
            return
