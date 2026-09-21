from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Union

from uav_vision.config.models import ModelSize


class CameraType(str, Enum):
    OPENCV = "opencv"
    GSTREAMER = "gstreamer"


class ProcessingType(str, Enum):
    DETECTION = "detection"


class LogLevel(str, Enum):
    BASIC = "basic"
    DEBUG = "debug"


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
class LogSettings:
    level: LogLevel = LogLevel.BASIC
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, LogLevel):
            raise ValueError("log level selection is invalid")
        if self.path is not None:
            if not isinstance(self.path, Path):
                raise ValueError("log path must be a pathlib.Path")
            if str(self.path).strip() in {"", "."}:
                raise ValueError("log path must not be empty")


@dataclass(frozen=True)
class YoloSettings:
    path: Optional[Path]
    origin: str
    models: Mapping[ProcessingType, Mapping[ModelSize, str]]

    def __post_init__(self) -> None:
        if self.path is not None and (
            not isinstance(self.path, Path) or str(self.path).strip() in {"", "."}
        ):
            raise ValueError("YOLO configuration path must be a non-empty pathlib.Path")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ValueError("YOLO configuration origin must be a non-empty string")
        for processing_type, model_paths in self.models.items():
            if not isinstance(processing_type, ProcessingType):
                raise ValueError("YOLO processing type selection is invalid")
            if not isinstance(model_paths, Mapping):
                raise ValueError("YOLO model mappings must be objects")
            for model_size, model_path in model_paths.items():
                if not isinstance(model_size, ModelSize):
                    raise ValueError("YOLO model size selection is invalid")
                if not isinstance(model_path, str) or not model_path.strip():
                    raise ValueError("YOLO model path must be a non-empty string")
        object.__setattr__(
            self,
            "models",
            MappingProxyType(
                {
                    processing_type: MappingProxyType(dict(model_paths))
                    for processing_type, model_paths in self.models.items()
                }
            ),
        )


@dataclass(frozen=True)
class AppSettings:
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    display: Optional[DisplaySettings] = None
    logging: LogSettings = field(default_factory=LogSettings)
    yolo: Optional[YoloSettings] = None

    def __post_init__(self) -> None:
        if not isinstance(self.capture, CaptureSettings):
            raise ValueError("capture settings are invalid")
        if not isinstance(self.processing, ProcessingSettings):
            raise ValueError("processing settings are invalid")
        if not isinstance(self.inference, InferenceSettings):
            raise ValueError("inference settings are invalid")
        if self.display is not None and not isinstance(self.display, DisplaySettings):
            raise ValueError("display settings are invalid")
        if not isinstance(self.logging, LogSettings):
            raise ValueError("log settings are invalid")
        if self.yolo is not None:
            if not isinstance(self.yolo, YoloSettings):
                raise ValueError("YOLO settings are invalid")
            if self.inference.model_size is not None:
                try:
                    self.yolo.models[self.processing.processing_type][
                        self.inference.model_size
                    ]
                except KeyError as error:
                    raise ValueError(
                        "YOLO configuration does not support the selected processing "
                        "type and model size"
                    ) from error

    @property
    def log_level(self) -> LogLevel:
        return self.logging.level

    @property
    def log_path(self) -> Optional[Path]:
        return self.logging.path
