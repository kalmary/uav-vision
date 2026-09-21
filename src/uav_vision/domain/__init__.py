from .depth import DepthResult
from .detection import BoundingBox, Detection, DetectionResult
from .frame import Frame, ProcessedFrame, ProcessingDiagnostics
from .segmentation import SegmentationClass, SegmentationResult

__all__ = (
    "BoundingBox",
    "DepthResult",
    "Detection",
    "DetectionResult",
    "Frame",
    "ProcessedFrame",
    "ProcessingDiagnostics",
    "SegmentationClass",
    "SegmentationResult",
)
