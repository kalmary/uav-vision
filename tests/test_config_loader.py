import json
import subprocess
import sys
import zipfile
from importlib import resources
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns, which

import pytest
import yaml

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

PACKAGED_NANO_MODELS = {
    "detection": ("models/yolo26n.pt", 640),
    "segmentation": ("models/yolo26n-sem.pt", 640),
    "depth": ("models/yolo26n-depth.pt", 768),
}


def assert_packaged_model_mappings(values):
    for processing_type, expected in PACKAGED_NANO_MODELS.items():
        model = values[processing_type]["nano"]
        assert (model["identifier"], model["input_size"]) == expected


def test_packaged_defaults_provide_basic_log_level_from_app_configuration():
    settings = load_settings()

    app_defaults = yaml.safe_load(
        resources.read_text("uav_vision.config.defaults", "app.yaml")
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


def test_genuine_yaml_configuration_resolves_a_relative_yolo_file(tmp_path):
    yolo_path = tmp_path / "models.yaml"
    yolo_path.write_text(
        """\
detection:
  nano:
    identifier: models/yaml-detector.engine
    input_size: 320
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """\
capture:
  source: 4
display:
  enabled: true
yolo_config_path: models.yaml
""",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.capture.source == 4
    assert settings.display.enabled is True
    assert settings.model.identifier == "models/yaml-detector.engine"
    assert settings.model.input_size == 320
    assert settings.yolo is not None
    assert settings.yolo.path == yolo_path
    assert settings.yolo.origin == str(yolo_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("capture: [", "Invalid YAML"),
        ("!!python/object/apply:os.system ['unsafe']", "Invalid YAML"),
        ("", "YAML mapping"),
        ("- item", "YAML mapping"),
        ("configuration", "YAML mapping"),
    ],
)
def test_configuration_rejects_invalid_yaml_and_non_mapping_roots(
    tmp_path, contents, message
):
    config_path = tmp_path / "app.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_settings(config_path)


@pytest.mark.parametrize(
    "content",
    [
        "1: invalid\nunknown: invalid\n",
        "capture:\n  1: invalid\n  unknown: invalid\n",
    ],
)
def test_configuration_rejects_non_string_yaml_keys(tmp_path, content):
    config_path = tmp_path / "app.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="keys.*strings"):
        load_settings(config_path)


def test_complete_user_configuration_replaces_packaged_defaults(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"small": {"identifier": "models/detector.engine"}}}),
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
    app_defaults = resources.read_text("uav_vision.config.defaults", "app.yaml")
    yolo_defaults = resources.read_text("uav_vision.config.defaults", "yolo.yaml")

    assert yaml.safe_load(app_defaults)["log_level"] == "basic"
    assert_packaged_model_mappings(yaml.safe_load(yolo_defaults))


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
        json.dumps({"detection": {"nano": {"identifier": "models/detector.engine"}}}),
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
        settings.yolo.models[ProcessingType.DETECTION][ModelSize.NANO].identifier
        == "models/detector.engine"
    )
    assert settings.model.input_size == 640


def test_loaded_yolo_model_mappings_are_immutable():
    settings = load_settings()

    assert settings.yolo is not None
    with pytest.raises(TypeError):
        settings.yolo.models[ProcessingType.DETECTION][ModelSize.NANO] = "other.pt"


def test_yolo_input_size_is_resolved_with_the_selected_model(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"small": {"input_size": 512}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps(
            {
                "inference": {"model_size": "small"},
                "yolo_config_path": "models.json",
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.model.identifier == "models/yolo26s.pt"
    assert settings.model.input_size == 512


def test_different_configured_model_sizes_resolve_as_atomic_selections(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps(
            {
                "detection": {
                    "small": {"input_size": 512},
                    "medium": {"input_size": 768},
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    small = load_settings(config_path, overrides={"inference": {"model_size": "small"}})
    medium = load_settings(
        config_path, overrides={"inference": {"model_size": "medium"}}
    )

    assert (small.model.identifier, small.model.input_size) == (
        "models/yolo26s.pt",
        512,
    )
    assert (medium.model.identifier, medium.model.input_size) == (
        "models/yolo26m.pt",
        768,
    )


@pytest.mark.parametrize(
    ("model_size", "identifier"),
    [
        ("nano", "models/yolo26n-sem.pt"),
        ("small", "models/yolo26s-sem.pt"),
        ("medium", "models/yolo26m-sem.pt"),
        ("large", "models/yolo26l-sem.pt"),
        ("xlarge", "models/yolo26x-sem.pt"),
    ],
)
def test_packaged_segmentation_models_resolve_with_expected_input_size(
    model_size, identifier
):
    settings = load_settings(
        overrides={
            "processing": {"processing_type": "segmentation"},
            "inference": {"model_size": model_size},
        }
    )

    assert settings.model.identifier == identifier
    assert settings.model.input_size == 640


@pytest.mark.parametrize(
    ("model_size", "identifier"),
    [
        ("nano", "models/yolo26n-depth.pt"),
        ("small", "models/yolo26s-depth.pt"),
        ("medium", "models/yolo26m-depth.pt"),
        ("large", "models/yolo26l-depth.pt"),
        ("xlarge", "models/yolo26x-depth.pt"),
    ],
)
def test_packaged_depth_models_resolve_with_expected_input_size(model_size, identifier):
    settings = load_settings(
        overrides={
            "processing": {"processing_type": "depth"},
            "inference": {"model_size": model_size},
        }
    )

    assert settings.model.identifier == identifier
    assert settings.model.input_size == 768


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
    assert settings.yolo.origin == "package:uav_vision.config.defaults/yolo.yaml"
    assert settings.yolo.detection_filters == DetectionFilterSettings()


def test_packaged_app_yolo_path_selects_the_named_packaged_configuration(
    monkeypatch,
):
    read_text = loader.resources.read_text

    def packaged_text(package, name):
        if name == "app.yaml":
            values = yaml.safe_load(read_text(package, name))
            values["yolo_config_path"] = "alternate-yolo.yaml"
            return yaml.safe_dump(values)
        if name == "alternate-yolo.yaml":
            return yaml.safe_dump(
                {
                    "detection": {
                        "nano": {"identifier": "alternate.pt", "input_size": 320}
                    }
                }
            )
        return read_text(package, name)

    monkeypatch.setattr(loader.resources, "read_text", packaged_text)

    settings = load_settings()

    assert settings.yolo is not None
    assert settings.yolo.path is None
    assert (
        settings.yolo.origin == "package:uav_vision.config.defaults/alternate-yolo.yaml"
    )
    assert (
        settings.yolo.models[ProcessingType.DETECTION][ModelSize.NANO].identifier
        == "alternate.pt"
    )


def test_packaged_yolo_model_without_input_size_is_rejected(monkeypatch):
    read_text = loader.resources.read_text

    def packaged_text(package, name):
        if name == "app.yaml":
            values = yaml.safe_load(read_text(package, name))
            values["yolo_config_path"] = "alternate-yolo.yaml"
            return yaml.safe_dump(values)
        if name == "alternate-yolo.yaml":
            return yaml.safe_dump(
                {"detection": {"nano": {"identifier": "alternate.pt"}}}
            )
        return read_text(package, name)

    monkeypatch.setattr(loader.resources, "read_text", packaged_text)

    with pytest.raises(ValueError, match="input_size"):
        load_settings()


def test_yolo_configuration_rejects_unknown_model_setting(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"nano": {"unsupported": True}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_settings(config_path)


def test_yolo_configuration_rejects_invalid_model_identifier(tmp_path):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"nano": {"identifier": ""}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="model identifier"):
        load_settings(config_path)


def test_explicit_yolo_path_is_resolved_relative_to_the_current_directory(
    tmp_path,
    monkeypatch,
):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"nano": {"identifier": "models/detector.engine"}}}),
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
        ("{", "Invalid YAML"),
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
        json.dumps({"detection": {"tiny": {"identifier": "detector.pt"}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_settings(config_path)


@pytest.mark.parametrize("input_size", [0, -1, 1.5, True, "640"])
def test_yolo_configuration_rejects_invalid_input_size(tmp_path, input_size):
    yolo_path = tmp_path / "models.json"
    yolo_path.write_text(
        json.dumps({"detection": {"nano": {"input_size": input_size}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "app.json"
    config_path.write_text(
        json.dumps({"yolo_config_path": "models.json"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="input size"):
        load_settings(config_path)


def test_built_wheel_contains_yaml_packaged_default_files(tmp_path):
    project_root = Path(__file__).parents[1]
    build_root = tmp_path / "project"
    copytree(
        project_root / "src",
        build_root / "src",
        ignore=ignore_patterns("*.egg-info", "__pycache__"),
    )
    copy2(project_root / "pyproject.toml", build_root / "pyproject.toml")
    uv_path = which("uv")
    assert uv_path is not None
    result = subprocess.run(
        [uv_path, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=build_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    wheel_path = next(tmp_path.glob("uav_vision-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        yolo_defaults = yaml.safe_load(
            wheel.read("uav_vision/config/defaults/yolo.yaml").decode("utf-8")
        )

    assert "uav_vision/config/defaults/app.yaml" in names
    assert "uav_vision/config/defaults/yolo.yaml" in names
    assert "uav_vision/config/defaults/app.json" not in names
    assert "uav_vision/config/defaults/yolo.json" not in names
    assert_packaged_model_mappings(yolo_defaults)

    installation = tmp_path / "installed"
    installed = subprocess.run(
        [
            uv_path,
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installation),
            str(wheel_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr

    isolated = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, yaml; "
                "sys.path.insert(0, sys.argv[1]); "
                "import uav_vision; "
                "from importlib import resources; "
                "assert sys.argv[1] in uav_vision.__file__; "
                "yaml.safe_load(resources.read_text("
                "'uav_vision.config.defaults', 'app.yaml')); "
                "yaml.safe_load(resources.read_text("
                "'uav_vision.config.defaults', 'yolo.yaml'))"
            ),
            str(installation),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert isolated.returncode == 0, isolated.stderr
