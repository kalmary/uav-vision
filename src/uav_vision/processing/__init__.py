from .base import FrameProcessor

__all__ = ("DetectionProcessor", "FrameProcessor")


def __getattr__(name):
    if name == "DetectionProcessor":
        from .detection import DetectionProcessor

        return DetectionProcessor
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
