import argparse
from pathlib import Path
from typing import Optional, Sequence

from uav_vision.config.models import ModelSize
from uav_vision.config.settings import (
    AppSettings,
    CameraSource,
    CameraType,
    CaptureSettings,
    DisplaySettings,
    InferenceSettings,
    ProcessingSettings,
    ProcessingType,
)


def _camera_source(value: str) -> CameraSource:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("camera source must not be empty")
    if value.lstrip("+-").isdigit():
        return int(value)
    return value


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _parser(*, prog: str, capture_defaults: CaptureSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "--camera-type",
        choices=CameraType,
        type=CameraType,
        default=capture_defaults.camera_type,
        metavar="{opencv,gstreamer}",
    )
    parser.add_argument(
        "--camera-source", type=_camera_source, default=capture_defaults.source
    )
    parser.add_argument(
        "--processing-type",
        choices=ProcessingType,
        type=ProcessingType,
        default=ProcessingType.DETECTION,
        metavar="{detection}",
    )
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--model-size",
        choices=ModelSize,
        type=ModelSize,
        metavar="{nano,small,medium,large,xlarge}",
    )
    model.add_argument("--model-path", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-width", type=_positive_int, default=1280)
    parser.add_argument("--display-height", type=_positive_int, default=720)
    return parser


def parse_args(
    arguments: Optional[Sequence[str]] = None,
    *,
    prog: str = "uav-vision",
    capture_defaults: Optional[CaptureSettings] = None,
) -> AppSettings:
    parser = _parser(
        prog=prog,
        capture_defaults=capture_defaults or CaptureSettings(),
    )
    values = parser.parse_args(arguments)
    model_size = values.model_size
    if model_size is None and values.model_path is None:
        model_size = ModelSize.NANO

    try:
        capture = CaptureSettings(values.camera_type, values.camera_source)
        processing = ProcessingSettings(values.processing_type)
        inference = InferenceSettings(model_size, values.model_path, values.device)
        display = None
        if values.display:
            display = DisplaySettings(values.display_width, values.display_height)
        return AppSettings(capture, processing, inference, display)
    except ValueError as error:
        parser.error(str(error))
