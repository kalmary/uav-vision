from typing import Protocol, runtime_checkable

from uav_vision.domain import ProcessedFrame


@runtime_checkable
class FrameOutput(Protocol):
    """Publish one processed frame and optionally request shutdown."""

    def write(self, processed: ProcessedFrame) -> bool: ...

    def close(self) -> None: ...
