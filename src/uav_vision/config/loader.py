import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional

from uav_vision.config.models import ModelSize
from uav_vision.config.settings import (
    AppSettings,
    CameraType,
    CaptureSettings,
    DisplaySettings,
    InferenceSettings,
    LogLevel,
    LogSettings,
    ProcessingSettings,
    ProcessingType,
    YoloSettings,
)

_DEFAULTS_PACKAGE = "uav_vision.config.defaults"


def _parse_json(text: str, origin: str) -> Mapping[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid JSON in configuration file: {}".format(origin)
        ) from error
    if not isinstance(data, dict):
        raise ValueError(
            "Configuration file must contain a JSON object: {}".format(origin)
        )
    return data


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            return _parse_json(stream.read(), str(path))
    except FileNotFoundError as error:
        raise ValueError(
            "Configuration file does not exist: {}".format(path)
        ) from error
    except OSError as error:
        raise ValueError(
            "Unable to read configuration file: {}".format(path)
        ) from error


def _packaged_origin(name: str) -> str:
    return "package:{}/{}".format(_DEFAULTS_PACKAGE, name)


def _packaged_resource_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("YOLO configuration path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("Packaged YOLO configuration path must be relative")
    return value


def _read_packaged_json(name: str) -> Mapping[str, Any]:
    origin = _packaged_origin(name)
    try:
        text = resources.read_text(_DEFAULTS_PACKAGE, name)
    except FileNotFoundError as error:
        raise ValueError(
            "Configuration file does not exist: {}".format(origin)
        ) from error
    return _parse_json(text, origin)


def _merge(
    defaults: Mapping[str, Any], values: Mapping[str, Any], context: str
) -> dict:
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(
            "Unknown configuration key in {}: {}".format(context, sorted(unknown)[0])
        )
    merged = dict(defaults)
    for key, value in values.items():
        default = defaults[key]
        if isinstance(value, dict) and isinstance(default, dict):
            merged[key] = _merge(default, value, context)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: Any, directory: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(label))
    path = Path(value)
    return path if path.is_absolute() else directory / path


def _app_settings(values: Mapping[str, Any], yolo: YoloSettings) -> AppSettings:
    capture_values = values["capture"]
    processing_values = values["processing"]
    inference_values = values["inference"]
    display_values = values["display"]
    if not isinstance(capture_values, dict):
        raise ValueError("capture configuration must be an object")
    if not isinstance(processing_values, dict):
        raise ValueError("processing configuration must be an object")
    if not isinstance(inference_values, dict):
        raise ValueError("inference configuration must be an object")
    if not isinstance(display_values, dict):
        raise ValueError("display configuration must be an object")
    if not isinstance(display_values.get("enabled"), bool):
        raise ValueError("display enabled must be a boolean")
    if (
        not isinstance(values.get("yolo_config_path"), str)
        or not values["yolo_config_path"].strip()
    ):
        raise ValueError("YOLO configuration path must be a non-empty string")

    try:
        capture = CaptureSettings(
            CameraType(capture_values["camera_type"]), capture_values["source"]
        )
        processing = ProcessingSettings(
            ProcessingType(processing_values["processing_type"])
        )
        model_size_value = inference_values["model_size"]
        model_size = None if model_size_value is None else ModelSize(model_size_value)
        model_path_value = inference_values["model_path"]
        model_path = None if model_path_value is None else Path(model_path_value)
        inference = InferenceSettings(
            model_size, model_path, inference_values["device"]
        )
        display = DisplaySettings(display_values["width"], display_values["height"])
        log_level = LogLevel(values["log_level"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Invalid application configuration: {}".format(error)
        ) from error

    log_path_value = values["log_path"]
    if log_path_value is not None:
        if not isinstance(log_path_value, str):
            raise ValueError("log path must be a string or null")
        if not log_path_value.strip():
            raise ValueError("log path must not be empty")
    logging = LogSettings(
        log_level, None if log_path_value is None else Path(log_path_value)
    )
    return AppSettings(
        capture,
        processing,
        inference,
        display if display_values["enabled"] else None,
        logging,
        yolo,
    )


def _yolo_settings(
    path: Optional[Path], origin: str, values: Mapping[str, Any]
) -> YoloSettings:
    models = {}
    try:
        for processing_name, model_values in values.items():
            if not isinstance(model_values, dict):
                raise ValueError("YOLO mode mappings must be objects")
            processing_type = ProcessingType(processing_name)
            models[processing_type] = {
                ModelSize(model_size): model_path
                for model_size, model_path in model_values.items()
            }
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid YOLO configuration: {}".format(error)) from error
    return YoloSettings(path, origin, models)


def _validate_app_layer(
    defaults: Mapping[str, Any],
    values: Mapping[str, Any],
    context: str,
    yolo: YoloSettings,
) -> None:
    try:
        _app_settings(_merge(defaults, values, context), yolo)
    except ValueError as error:
        raise ValueError("Invalid {}: {}".format(context, error)) from error


def load_settings(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> AppSettings:
    app_defaults = _read_packaged_json("app.json")
    yolo_resource = _packaged_resource_name(app_defaults.get("yolo_config_path"))
    yolo_defaults = _read_packaged_json(yolo_resource)
    default_yolo = _yolo_settings(None, _packaged_origin(yolo_resource), yolo_defaults)
    _validate_app_layer(app_defaults, {}, "packaged defaults", default_yolo)

    selected_path = None if config_path is None else Path(config_path)
    if selected_path is not None:
        user_values = _read_json(selected_path)
        _validate_app_layer(
            app_defaults, user_values, "user configuration", default_yolo
        )
        app_values = _merge(app_defaults, user_values, str(selected_path))
    else:
        user_values = {}
        app_values = dict(app_defaults)

    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("Configuration overrides must be a mapping")
        _validate_app_layer(app_defaults, overrides, "explicit overrides", default_yolo)
        app_values = _merge(app_values, overrides, "explicit overrides")

    if overrides is not None and "yolo_config_path" in overrides:
        yolo_directory = Path.cwd()
    elif "yolo_config_path" in user_values:
        yolo_directory = selected_path.parent
    else:
        return _app_settings(app_values, default_yolo)
    yolo_path = _resolve_path(
        app_values["yolo_config_path"], yolo_directory, "YOLO configuration path"
    )
    yolo_values = _merge(yolo_defaults, _read_json(yolo_path), str(yolo_path))
    return _app_settings(
        app_values, _yolo_settings(yolo_path, str(yolo_path), yolo_values)
    )
