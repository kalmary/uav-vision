import json
import subprocess
import zipfile
from importlib import resources
from pathlib import Path
from shutil import which

import pytest

from uav_vision.config import loader
from uav_vision.config.loader import load_settings
from uav_vision.config.models import ModelSize
from uav_vision.config.settings import (
    AppSettings,
    DetectionFilterSettings,
    LogLevel,
    LogSettings,
    ProcessingType,
)


def test_packaged_defaults_provide_basic_log_level_from_app_configuration():
    settings = load_settings()

    app_defaults = json.loads(
        resources.read_text("uav_vision.config.defaults", "app.json")
    )

    assert app_defaults["log_level"] == "basic"
    assert settings.log_level is LogLevel.BASIC


def test_partial_user_configuration_preserves_display_defaults(tmp_path):
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"display": {"enabled": True, "width": 960}}),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.display is not None
    assert settings.display.width == 960
    assert settings.display.height == 720


def test_complete_user_configuration_replaces_packaged_defaults(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"small": "models/detector.engine"}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps(
            {
                "capture": {"camera_type": "opencv", "source": 4},
                "processing": {"processing_type": "detection"},
                "inference": {
                    "model_size": "small",
                    "device": "cpu",
                },
                "display": {"enabled": True, "width": 960, "height": 540},
                "log_level": "debug",
                "log_path": "vision.log",
                "yolo_config_path": "models.json",
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.capture.source == 4
    assert settings.inference.model_size is ModelSize.SMALL
    assert settings.inference.device == "cpu"
    assert settings.display is not None
    assert settings.display.width == 960
    assert settings.display.height == 540
    assert settings.log_level is LogLevel.DEBUG
    assert settings.log_path == Path("vision.log")
    assert settings.yolo is not None
    assert settings.yolo.path == yolo_path
    assert settings.yolo.origin == str(yolo_path)


def test_editable_package_resources_expose_both_default_files():
    app_defaults = resources.read_text("uav_vision.config.defaults", "app.json")
    yolo_defaults = resources.read_text("uav_vision.config.defaults", "yolo.json")

    assert json.loads(app_defaults)["log_level"] == "basic"
    assert "detection" in json.loads(yolo_defaults)


def test_user_configuration_rejects_non_boolean_display_enabled(tmp_path):
    config_path = tmp_path / "app.json"
    config_path.write_text(json.dumps({"display": {"enabled": 1}}), encoding="utf-8")

    with pytest.raises(ValueError, match="display enabled"):
        load_settings(config_path)


def test_explicit_overrides_take_priority_over_a_selected_configuration(tmp_path):
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps(
            {
                "capture": {"source": 2},
                "inference": {"model_size": "small"},
                "log_level": "debug",
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        config_path,
        overrides={
            "capture": {"source": 3},
            "inference": {"model_size": "medium"},
            "log_level": "basic",
        },
    )

    assert settings.capture.source == 3
    assert settings.inference.model_size is ModelSize.MEDIUM
    assert settings.log_level is LogLevel.BASIC


def test_invalid_user_model_size_is_not_masked_by_an_explicit_override(tmp_path):
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"inference": {"model_size": "invalid"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid user configuration"):
        load_settings(config_path, overrides={"inference": {"model_size": "small"}})


def test_user_object_replaced_by_a_scalar_is_not_masked_by_an_explicit_override(
    tmp_path,
):
    config_path = tmp_path / "app.json"
    config_path.write_text(json.dumps({"capture": "camera"}), encoding="utf-8")

    with pytest.raises(ValueError, match="capture configuration"):
        load_settings(
            config_path,
            overrides={"capture": {"camera_type": "opencv", "source": 1}},
        )


def test_yolo_path_is_resolved_relative_to_the_selected_application_file(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"nano": "models/detector.engine"}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    settings = load_settings(config_path)

    assert settings.yolo is not None
    assert settings.yolo.path == yolo_path
    assert (
        settings.yolo.models[ProcessingType.DETECTION][ModelSize.NANO]
        == "models/detector.engine"
    )


def test_loaded_yolo_model_mappings_are_immutable():
    settings = load_settings()

    assert settings.yolo is not None
    with pytest.raises(TypeError):
        settings.yolo.models[ProcessingType.DETECTION][ModelSize.NANO] = "other.pt"


def test_yolo_detection_filters_are_loaded_as_immutable_settings(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps(
            {
                "detection_filters": {
                    "selected_classes": [2, 5],
                    "minimum_confidence": 0.6,
                    "top_k": 3,
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    settings = load_settings(config_path)

    assert settings.yolo is not None
    assert settings.yolo.detection_filters == DetectionFilterSettings((2, 5), 0.6, 3)


def test_packaged_defaults_are_read_without_a_temporary_resource_path(monkeypatch):
    def unexpected_resource_path(*args, **kwargs):
        raise AssertionError("loader must not expose a temporary resource path")

    monkeypatch.setattr(loader.resources, "path", unexpected_resource_path)

    settings = load_settings()

    assert settings.yolo is not None
    assert settings.yolo.path is None
    assert settings.yolo.origin == "package:uav_vision.config.defaults/yolo.json"
    assert settings.yolo.detection_filters == DetectionFilterSettings()


def test_packaged_app_yolo_path_selects_the_named_packaged_configuration(
    monkeypatch,
):
    read_text = loader.resources.read_text

    def packaged_text(package, name):
        if name == "app.json":
            values = json.loads(read_text(package, name))
            values["yolo_config_path"] = "alternate-yolo.json"
            return json.dumps(values)
        if name == "alternate-yolo.json":
            return json.dumps({"detection": {"nano": "alternate.pt"}})
        return read_text(package, name)

    monkeypatch.setattr(loader.resources, "read_text", packaged_text)

    settings = load_settings()

    assert settings.yolo is not None
    assert settings.yolo.path is None
    assert (
        settings.yolo.origin == "package:uav_vision.config.defaults/alternate-yolo.json"
    )
    assert (
        settings.yolo.models[ProcessingType.DETECTION][ModelSize.NANO] == "alternate.pt"
    )


def test_explicit_yolo_path_is_resolved_relative_to_the_current_directory(
    tmp_path,
    monkeypatch,
):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"nano": "models/detector.engine"}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(overrides={"yolo_config_path": "models.json"})

    assert settings.yolo is not None
    assert settings.yolo.path == yolo_path
    assert settings.yolo.origin == str(yolo_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{", "Invalid JSON"),
        (json.dumps({"unknown": True}), "Unknown configuration key"),
        (json.dumps({"capture": {"source": True}}), "camera source"),
        (json.dumps({"inference": {"model_size": "unknown"}}), "Invalid application"),
    ],
)
def test_invalid_user_configuration_fails_descriptively(tmp_path, contents, message):
    config_path = tmp_path / "app.json"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_settings(config_path)


def test_missing_user_configuration_fails_descriptively(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_settings(tmp_path / "missing.json")


def test_configuration_rejects_an_empty_log_path(tmp_path):
    config_path = tmp_path / "app.json"
    config_path.write_text(json.dumps({"log_path": "   "}), encoding="utf-8")

    with pytest.raises(ValueError, match="log path"):
        load_settings(config_path)


def test_log_settings_rejects_an_empty_path():
    with pytest.raises(ValueError, match="log path"):
        LogSettings(path=Path("   "))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture", "capture"),
        ("processing", "processing"),
        ("inference", "inference"),
        ("display", "display"),
    ],
)
def test_app_settings_rejects_invalid_component_types(field, value):
    with pytest.raises(ValueError, match="{} settings".format(field)):
        AppSettings(**{field: value})


def test_missing_yolo_configuration_fails_descriptively(tmp_path):
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "missing.json"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not exist"):
        load_settings(config_path)


def test_yolo_configuration_rejects_unknown_model_size(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"tiny": "detector.pt"}}), encoding="utf-8"
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_settings(config_path)


def test_built_wheel_contains_both_packaged_default_files(tmp_path):
    project_root = Path(__file__).parents[1]
    uv_path = which("uv")
    assert uv_path is not None
    result = subprocess.run(
        [uv_path, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    wheel_path = next(tmp_path.glob("uav_vision-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    assert "uav_vision/config/defaults/app.json" in names
    assert "uav_vision/config/defaults/yolo.json" in names
