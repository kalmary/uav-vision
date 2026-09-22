from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union

from uav_vision.config.models import ModelSize


class CameraType(str, Enum):
    OPENCV = "opencv"
    GSTREAMER = "gstreamer"


class ProcessingType(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"


@dataclass(frozen=True)
class DetectionFilterSettings:
    selected_classes: Tuple[int, ...] = ()
    minimum_confidence: float = 0.0
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.selected_classes, tuple) or any(
            isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0
            for class_id in self.selected_classes
        ):
            raise ValueError("selected classes must be non-negative integers")
        if (
            isinstance(self.minimum_confidence, bool)
            or not isinstance(self.minimum_confidence, Real)
            or not isfinite(self.minimum_confidence)
            or not 0 <= self.minimum_confidence <= 1
        ):
            raise ValueError("minimum confidence must be a finite probability")
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ValueError("top-k must be a positive integer or null")


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
    model_size: ModelSize = ModelSize.NANO
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.model_size, ModelSize):
            raise ValueError("model size selection is invalid")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("inference device must be cpu or cuda")


@dataclass(frozen=True)
class DisplaySettings:
    width: int = 1280
    height: int = 720
    enabled: bool = False

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
        if not isinstance(self.enabled, bool):
            raise ValueError("display enabled must be a boolean")


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
    detection_filters: DetectionFilterSettings = field(
        default_factory=DetectionFilterSettings
    )

    def __post_init__(self) -> None:
        if self.path is not None and (
            not isinstance(self.path, Path) or str(self.path).strip() in {"", "."}
        ):
            raise ValueError("YOLO configuration path must be a non-empty pathlib.Path")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ValueError("YOLO configuration origin must be a non-empty string")
        if not isinstance(self.detection_filters, DetectionFilterSettings):
            raise ValueError("YOLO detection filters are invalid")
        for processing_type, model_identifiers in self.models.items():
            if not isinstance(processing_type, ProcessingType):
                raise ValueError("YOLO processing type selection is invalid")
            if not isinstance(model_identifiers, Mapping):
                raise ValueError("YOLO model mappings must be objects")
            for model_size, model_identifier in model_identifiers.items():
                if not isinstance(model_size, ModelSize):
                    raise ValueError("YOLO model size selection is invalid")
                if (
                    not isinstance(model_identifier, str)
                    or not model_identifier.strip()
                ):
                    raise ValueError("YOLO model identifier must be a non-empty string")
        object.__setattr__(
            self,
            "models",
            MappingProxyType(
                {
                    processing_type: MappingProxyType(dict(model_identifiers))
                    for processing_type, model_identifiers in self.models.items()
                }
            ),
        )


@dataclass(frozen=True)
class AppSettings:
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    logging: LogSettings = field(default_factory=LogSettings)
    yolo: Optional[YoloSettings] = None
    fps: int = 30

    def __post_init__(self) -> None:
        if not isinstance(self.capture, CaptureSettings):
            raise ValueError("capture settings are invalid")
        if not isinstance(self.processing, ProcessingSettings):
            raise ValueError("processing settings are invalid")
        if not isinstance(self.inference, InferenceSettings):
            raise ValueError("inference settings are invalid")
        if not isinstance(self.display, DisplaySettings):
            raise ValueError("display settings are invalid")
        if not isinstance(self.logging, LogSettings):
            raise ValueError("log settings are invalid")
        if self.yolo is not None and not isinstance(self.yolo, YoloSettings):
            raise ValueError("YOLO settings are invalid")
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps <= 0:
            raise ValueError("fps must be a positive integer")

    @property
    def model_identifier(self) -> str:
        if self.yolo is None:
            raise ValueError("YOLO configuration is required to resolve a model")
        try:
            return self.yolo.models[self.processing.processing_type][
                self.inference.model_size
            ]
        except KeyError as error:
            raise ValueError(
                "YOLO configuration does not support the selected processing type and "
                "model size"
            ) from error

    @property
    def log_level(self) -> LogLevel:
        return self.logging.level

    @property
    def log_path(self) -> Optional[Path]:
        return self.logging.path
