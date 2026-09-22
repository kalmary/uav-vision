import json
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional

from uav_vision.config.models import ModelSize
from uav_vision.config.settings import (
    AppSettings,
    CameraType,
    CaptureSettings,
    DetectionFilterSettings,
    DisplaySettings,
    InferenceSettings,
    LogLevel,
    LogSettings,
    ProcessingSettings,
    ProcessingType,
    YoloSettings,
)

_DEFAULTS_PACKAGE = "uav_vision.config.defaults"


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


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
        inference = InferenceSettings(
            ModelSize(inference_values["model_size"]), inference_values["device"]
        )
        display = DisplaySettings(
            display_values["width"], display_values["height"], display_values["enabled"]
        )
        log_level = LogLevel(values["log_level"])
        fps = values["fps"]
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
        display,
        logging,
        yolo,
        fps,
    )


def _yolo_settings(
    path: Optional[Path], origin: str, values: Mapping[str, Any]
) -> YoloSettings:
    models = {}
    try:
        for processing_name, model_values in values.items():
            if processing_name == "detection_filters":
                continue
            if not isinstance(model_values, dict):
                raise ValueError("YOLO mode mappings must be objects")
            processing_type = ProcessingType(processing_name)
            models[processing_type] = {
                ModelSize(model_size): model_identifier
                for model_size, model_identifier in model_values.items()
            }
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid YOLO configuration: {}".format(error)) from error
    filter_values = values.get("detection_filters", {})
    if not isinstance(filter_values, dict):
        raise ValueError(
            "Invalid YOLO configuration: detection filters must be an object"
        )
    try:
        selected_classes = filter_values.get("selected_classes", [])
        if not isinstance(selected_classes, list):
            raise ValueError("selected classes must be an array")
        filters = DetectionFilterSettings(
            tuple(selected_classes),
            filter_values.get("minimum_confidence", 0.0),
            filter_values.get("top_k"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid YOLO configuration: {}".format(error)) from error
    return YoloSettings(path, origin, models, filters)


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


def _validate_resolved(
    settings: AppSettings, validate_model: bool, check_cuda: bool
) -> AppSettings:
    if validate_model:
        try:
            settings.model_identifier
        except ValueError as error:
            raise ValueError(
                "Invalid resolved configuration: {}".format(error)
            ) from error
    if check_cuda and settings.inference.device == "cuda" and not cuda_available():
        raise ValueError("CUDA device is unavailable")
    return settings


def _override_mapping(
    values: Mapping[str, Any], name: str, keys: set
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError("{} overrides must be an object".format(name))
    unknown = set(values) - keys
    if unknown:
        raise ValueError("Unknown {} override: {}".format(name, sorted(unknown)[0]))
    return values


def apply_overrides(
    settings: AppSettings,
    overrides: Optional[Mapping[str, Any]] = None,
    yolo_overrides: Optional[Mapping[str, Any]] = None,
) -> AppSettings:
    if not isinstance(settings, AppSettings):
        raise ValueError("base settings are invalid")
    values = {} if overrides is None else _override_mapping(
        overrides,
        "configuration",
        {
            "capture",
            "processing",
            "inference",
            "display",
            "log_level",
            "log_path",
            "fps",
        },
    )
    try:
        capture_values = _override_mapping(
            values.get("capture", {}), "capture", {"camera_type", "source"}
        )
        capture = CaptureSettings(
            CameraType(capture_values.get("camera_type", settings.capture.camera_type)),
            capture_values.get("source", settings.capture.source),
        )
        processing_values = _override_mapping(
            values.get("processing", {}), "processing", {"processing_type"}
        )
        processing = ProcessingSettings(
            ProcessingType(
                processing_values.get(
                    "processing_type", settings.processing.processing_type
                )
            )
        )
        inference_values = _override_mapping(
            values.get("inference", {}), "inference", {"model_size", "device"}
        )
        inference = InferenceSettings(
            ModelSize(inference_values.get("model_size", settings.inference.model_size)),
            inference_values.get("device", settings.inference.device),
        )
        display_values = _override_mapping(
            values.get("display", {}), "display", {"enabled", "width", "height"}
        )
        display = DisplaySettings(
            display_values.get("width", settings.display.width),
            display_values.get("height", settings.display.height),
            display_values.get("enabled", settings.display.enabled),
        )
        level = LogLevel(values.get("log_level", settings.logging.level))
        path_value = values.get("log_path", settings.logging.path)
        logging = LogSettings(level, None if path_value is None else Path(path_value))
        resolved = replace(
            settings,
            capture=capture,
            processing=processing,
            inference=inference,
            display=display,
            logging=logging,
            fps=values.get("fps", settings.fps),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid explicit overrides: {}".format(error)) from error

    if yolo_overrides is not None:
        values = _override_mapping(yolo_overrides, "YOLO", {"detection_filters"})
        filter_values = _override_mapping(
            values.get("detection_filters", {}),
            "detection filter",
            {"selected_classes", "minimum_confidence", "top_k"},
        )
        filters = resolved.yolo.detection_filters
        try:
            filters = DetectionFilterSettings(
                tuple(filter_values.get("selected_classes", filters.selected_classes)),
                filter_values.get("minimum_confidence", filters.minimum_confidence),
                filter_values.get("top_k", filters.top_k),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid explicit YOLO overrides: {}".format(error)) from error
        resolved = replace(
            resolved,
            yolo=YoloSettings(
                resolved.yolo.path,
                resolved.yolo.origin,
                resolved.yolo.models,
                filters,
            ),
        )
    return _validate_resolved(resolved, validate_model=True, check_cuda=True)


def load_settings(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    yolo_overrides: Optional[Mapping[str, Any]] = None,
    validate_model: bool = True,
    check_cuda: bool = True,
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
        yolo_path = _resolve_path(
            app_values["yolo_config_path"], Path.cwd(), "YOLO configuration path"
        )
    elif "yolo_config_path" in user_values:
        yolo_path = _resolve_path(
            app_values["yolo_config_path"],
            selected_path.parent,
            "YOLO configuration path",
        )
    else:
        yolo_path = None

    if yolo_path is None:
        yolo_values = dict(yolo_defaults)
        yolo = default_yolo
    else:
        yolo_values = _merge(yolo_defaults, _read_json(yolo_path), str(yolo_path))
        yolo = _yolo_settings(yolo_path, str(yolo_path), yolo_values)

    if yolo_overrides is not None:
        if not isinstance(yolo_overrides, Mapping):
            raise ValueError("YOLO overrides must be a mapping")
        yolo_values = _merge(yolo_values, yolo_overrides, "explicit YOLO overrides")
        yolo = _yolo_settings(
            yolo.path,
            yolo.origin,
            yolo_values,
        )

    return _validate_resolved(
        _app_settings(app_values, yolo), validate_model, check_cuda
    )
