from typing import Protocol, Tuple, runtime_checkable

from uav_vision.domain import Detection, Frame


class InferenceError(RuntimeError):
    """Base class for inference failures."""


class InferenceInitializationError(InferenceError):
    """Raised when an inference model cannot be loaded."""


class InferenceRunError(InferenceError):
    """Raised when an inference provider cannot process a frame."""


class InferenceResultError(InferenceError):
    """Raised when an inference provider returns malformed results."""


@runtime_checkable
class Detector(Protocol):
    """Detect application-owned objects in one frame."""

    def detect(self, frame: Frame) -> Tuple[Detection, ...]: ...
