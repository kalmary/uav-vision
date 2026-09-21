from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from uav_vision.config.models import ModelSize


class CameraType(str, Enum):
    OPENCV = "opencv"
    GSTREAMER = "gstreamer"


class ProcessingType(str, Enum):
    DETECTION = "detection"


CameraSource = Union[int, str]


@dataclass(frozen=True)
class CaptureSettings:
    camera_type: CameraType = CameraType.OPENCV
    source: CameraSource = 0

    def __post_init__(self) -> None:
        if not isinstance(self.camera_type, CameraType):
            raise ValueError("camera type selection is invalid")
        if isinstance(self.source, bool):
            raise ValueError("camera source must be a non-negative index or string")
        if isinstance(self.source, int):
            if self.source < 0:
                raise ValueError("camera source index must be non-negative")
            if self.camera_type is CameraType.GSTREAMER:
                raise ValueError("GStreamer camera source must be a pipeline string")
            return
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("camera source must be a non-empty string")


@dataclass(frozen=True)
class ProcessingSettings:
    processing_type: ProcessingType = ProcessingType.DETECTION

    def __post_init__(self) -> None:
        if not isinstance(self.processing_type, ProcessingType):
            raise ValueError("processing type selection is invalid")


@dataclass(frozen=True)
class InferenceSettings:
    model_size: Optional[ModelSize] = ModelSize.NANO
    model_path: Optional[Path] = None
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model_size is not None and not isinstance(self.model_size, ModelSize):
            raise ValueError("model size selection is invalid")
        if self.model_path is not None and not isinstance(self.model_path, Path):
            raise ValueError("model path must be a pathlib.Path")
        if (self.model_size is None) == (self.model_path is None):
            raise ValueError("select either a model size or an explicit model path")
        if self.model_path is not None and str(self.model_path).strip() in {"", "."}:
            raise ValueError("model path must not be empty")
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device.strip()
        ):
            raise ValueError("inference device must not be empty")


@dataclass(frozen=True)
class DisplaySettings:
    width: int = 1280
    height: int = 720

    def __post_init__(self) -> None:
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("display dimensions must be positive integers")


@dataclass(frozen=True)
class AppSettings:
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    display: Optional[DisplaySettings] = None
