import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from uav_vision.config.loader import apply_overrides, load_settings
from uav_vision.config.models import ModelSize
from uav_vision.config.settings import AppSettings, CameraType, LogLevel, ProcessingType


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        ) from error
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def _confidence(value: str) -> float:
    try:
        confidence = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("confidence must be a number") from error
    if not 0 <= confidence <= 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return confidence


def _bootstrap_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, add_help=False)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument(
        "--processing-type", choices=ProcessingType, type=ProcessingType
    )
    return parser


def _parser(*, prog: str, processing_type: ProcessingType) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    configuration = parser.add_argument_group("configuration")
    configuration.add_argument("--config-path", type=Path)

    camera = parser.add_argument_group("camera")
    camera.add_argument(
        "--camera-type",
        choices=CameraType,
        type=CameraType,
        metavar="{opencv,gstreamer}",
        default=argparse.SUPPRESS,
    )
    camera.add_argument(
        "--camera-index", type=_non_negative_int, default=argparse.SUPPRESS
    )

    inference = parser.add_argument_group("processing/inference")
    inference.add_argument(
        "--processing-type",
        choices=ProcessingType,
        type=ProcessingType,
        metavar="{detection,segmentation,depth}",
        default=argparse.SUPPRESS,
    )
    inference.add_argument(
        "--model-size",
        choices=ModelSize,
        type=ModelSize,
        metavar="{nano,small,medium,large,xlarge}",
        default=argparse.SUPPRESS,
    )
    inference.add_argument(
        "--device", choices=("cpu", "cuda"), default=argparse.SUPPRESS
    )
    if processing_type is ProcessingType.DETECTION:
        inference.add_argument(
            "--selected-classes",
            type=_non_negative_int,
            nargs="+",
            default=argparse.SUPPRESS,
        )
        inference.add_argument(
            "--minimum-confidence", type=_confidence, default=argparse.SUPPRESS
        )
        inference.add_argument("--top-k", type=_positive_int, default=argparse.SUPPRESS)

    display = parser.add_argument_group("display")
    display_state = display.add_mutually_exclusive_group()
    display_state.add_argument(
        "--display",
        action="store_true",
        dest="display_enabled",
        default=argparse.SUPPRESS,
    )
    display_state.add_argument(
        "--no-display",
        action="store_false",
        dest="display_enabled",
        default=argparse.SUPPRESS,
    )
    display.add_argument(
        "--display-width", type=_positive_int, default=argparse.SUPPRESS
    )
    display.add_argument(
        "--display-height", type=_positive_int, default=argparse.SUPPRESS
    )

    logging = parser.add_argument_group("logging")
    logging.add_argument(
        "--log-level", choices=LogLevel, type=LogLevel, default=argparse.SUPPRESS
    )
    logging.add_argument("--log-path", type=Path, default=argparse.SUPPRESS)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--fps", type=_positive_int, default=argparse.SUPPRESS)
    return parser


def _overrides(values: argparse.Namespace) -> Mapping[str, Any]:
    arguments = vars(values)
    overrides = {}
    capture = {}
    if "camera_type" in arguments:
        capture["camera_type"] = arguments["camera_type"].value
    if "camera_index" in arguments:
        capture["source"] = arguments["camera_index"]
    if capture:
        overrides["capture"] = capture
    if "processing_type" in arguments:
        overrides["processing"] = {
            "processing_type": arguments["processing_type"].value
        }
    inference = {}
    if "model_size" in arguments:
        inference["model_size"] = arguments["model_size"].value
    if "device" in arguments:
        inference["device"] = arguments["device"]
    if inference:
        overrides["inference"] = inference
    display = {}
    if "display_enabled" in arguments:
        display["enabled"] = arguments["display_enabled"]
    if "display_width" in arguments:
        display["width"] = arguments["display_width"]
    if "display_height" in arguments:
        display["height"] = arguments["display_height"]
    if display:
        overrides["display"] = display
    if "log_level" in arguments:
        overrides["log_level"] = arguments["log_level"].value
    if "log_path" in arguments:
        overrides["log_path"] = str(arguments["log_path"])
    if "fps" in arguments:
        overrides["fps"] = arguments["fps"]
    return overrides


def _yolo_overrides(values: argparse.Namespace) -> Mapping[str, Any]:
    arguments = vars(values)
    filters = {}
    for name in ("selected_classes", "minimum_confidence", "top_k"):
        if name in arguments:
            filters[name] = arguments[name]
    return {} if not filters else {"detection_filters": filters}


def parse_args(
    arguments: Optional[Sequence[str]] = None,
    *,
    prog: str = "uav-vision",
) -> AppSettings:
    bootstrap_parser = _bootstrap_parser(prog)
    bootstrap, _ = bootstrap_parser.parse_known_args(arguments)
    try:
        configured = load_settings(
            bootstrap.config_path,
            validate_model=False,
            check_cuda=False,
        )
    except ValueError as error:
        bootstrap_parser.error(str(error))
    processing_type = bootstrap.processing_type or configured.processing.processing_type
    parser = _parser(prog=prog, processing_type=processing_type)
    values = parser.parse_args(arguments)
    selected_type = getattr(values, "processing_type", processing_type)
    if selected_type is not ProcessingType.DETECTION:
        message = "processing type '{}' is unavailable: ".format(selected_type.value)
        parser.error(message + "no model or processor is configured")
    try:
        return apply_overrides(
            configured,
            _overrides(values),
            _yolo_overrides(values),
        )
    except ValueError as error:
        parser.error(str(error))
