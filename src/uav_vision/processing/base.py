from typing import Protocol, runtime_checkable

from uav_vision.domain import Frame, ProcessedFrame


@runtime_checkable
class FrameProcessor(Protocol):
    """Process one application-owned frame."""

    def process(self, frame: Frame) -> ProcessedFrame: ...
