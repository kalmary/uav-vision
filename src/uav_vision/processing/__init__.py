from .base import FrameProcessor

__all__ = (
    "DepthProcessor",
    "DetectionProcessor",
    "FrameProcessor",
    "SegmentationProcessor",
)


def __getattr__(name):
    if name == "DepthProcessor":
        from .depth import DepthProcessor

        return DepthProcessor
    if name == "DetectionProcessor":
        from .detection import DetectionProcessor

        return DetectionProcessor
    if name == "SegmentationProcessor":
        from .segmentation import SegmentationProcessor

        return SegmentationProcessor
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
