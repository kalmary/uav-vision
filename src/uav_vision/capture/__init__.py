from .base import (
    CaptureError,
    CaptureInitializationError,
    CaptureReadError,
    FrameSource,
)

__all__ = (
    "CaptureError",
    "CaptureInitializationError",
    "CaptureReadError",
    "FrameSource",
    "GStreamerCamera",
    "OpenCvCamera",
)


def __getattr__(name):
    if name == "GStreamerCamera":
        from .gstreamer import GStreamerCamera

        return GStreamerCamera
    if name == "OpenCvCamera":
        from .opencv import OpenCvCamera

        return OpenCvCamera
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
