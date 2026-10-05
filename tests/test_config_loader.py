import json
import subprocess
import sys
import zipfile
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

DEFAULT_NANO_MODELS = {
    "detection": ("models/yolo26n.pt", 640),
    "segmentation": ("models/yolo26n-sem.pt", 640),
    "depth": ("models/yolo26n-depth.pt", 768),
}


def assert_default_model_mappings(values):
    for processing_type, expected in DEFAULT_NANO_MODELS.items():
        model = values[processing_type]["nano"]
        assert (model["identifier"], model["input_size"]) == expected


def test_default_configuration_path_points_to_the_project_config_directory():
    project_root = Path(__file__).parents[1]

    assert loader._default_config_path() == project_root / "config" / "app.yaml"


def test_installed_default_configuration_path_uses_distribution_metadata(
    tmp_path, monkeypatch
):
    installation = tmp_path / "environment"
    installed_loader = installation / "lib" / "uav_vision" / "config" / "loader.py"
    config_path = installation / "share" / "uav-vision" / "config" / "app.yaml"

    class Distribution:
        files = (
            Path("share/uav-vision/config/app.yaml.backup"),
            Path("share/uav-vision/config/app.yaml"),
        )

        def locate_file(self, file):
            return installation / file

    monkeypatch.setattr(loader, "__file__", str(installed_loader))
    monkeypatch.setattr(loader.metadata, "distribution", lambda name: Distribution())

    assert loader._default_config_path() == config_path


def test_installed_default_configuration_requires_distribution_metadata(
    tmp_path, monkeypatch
):
    installed_loader = (
        tmp_path / "environment" / "lib" / "uav_vision" / "config" / "loader.py"
    )

    def missing_distribution(name):
        raise loader.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(loader, "__file__", str(installed_loader))
    monkeypatch.setattr(loader.metadata, "distribution", missing_distribution)

    with pytest.raises(ValueError, match="Unable to locate the default configuration"):
        loader._default_config_path()


def test_packaged_defaults_provide_basic_log_level_from_app_configuration():
    settings = load_settings()
    config_directory = Path(__file__).parents[1] / "config"

    app_defaults = yaml.safe_load(
        (config_directory / "app.yaml").read_text(encoding="utf-8")
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
    assert settings.yolo is not None
    assert settings.yolo.path == Path(__file__).parents[1] / "config" / "yolo.yaml"


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


def test_project_config_directory_exposes_both_default_files():
    config_directory = Path(__file__).parents[1] / "config"
    app_defaults = (config_directory / "app.yaml").read_text(encoding="utf-8")
    yolo_defaults = (config_directory / "yolo.yaml").read_text(encoding="utf-8")

    assert yaml.safe_load(app_defaults)["log_level"] == "basic"
    assert_default_model_mappings(yaml.safe_load(yolo_defaults))


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


def test_default_settings_are_read_outside_the_project_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    default_yolo_path = Path(__file__).parents[1] / "config" / "yolo.yaml"
    assert settings.yolo is not None
    assert settings.yolo.path == default_yolo_path
    assert settings.yolo.origin == str(default_yolo_path)
    assert settings.yolo.detection_filters == DetectionFilterSettings()


def test_default_app_yolo_path_selects_the_named_default_configuration(
    tmp_path, monkeypatch
):
    config_directory = Path(__file__).parents[1] / "config"
    app_values = yaml.safe_load(
        (config_directory / "app.yaml").read_text(encoding="utf-8")
    )
    app_values["capture"]["source"] = 7
    app_values["yolo_config_path"] = "alternate-yolo.yaml"
    app_path = tmp_path / "app.yaml"
    app_path.write_text(yaml.safe_dump(app_values), encoding="utf-8")
    yolo_values = yaml.safe_load(
        (config_directory / "yolo.yaml").read_text(encoding="utf-8")
    )
    yolo_values["detection"]["nano"] = {
        "identifier": "alternate.pt",
        "input_size": 320,
    }
    yolo_path = tmp_path / "alternate-yolo.yaml"
    yolo_path.write_text(yaml.safe_dump(yolo_values), encoding="utf-8")
    monkeypatch.setattr(loader, "_default_config_path", lambda: app_path)

    settings = load_settings()

    assert settings.capture.source == 7
    assert settings.yolo is not None
    assert settings.yolo.path == yolo_path
    assert settings.yolo.origin == str(yolo_path)
    assert (
        settings.yolo.models[ProcessingType.DETECTION][ModelSize.NANO].identifier
        == "alternate.pt"
    )


def test_default_yolo_model_without_input_size_is_rejected(tmp_path, monkeypatch):
    config_directory = Path(__file__).parents[1] / "config"
    app_values = yaml.safe_load(
        (config_directory / "app.yaml").read_text(encoding="utf-8")
    )
    app_values["yolo_config_path"] = "invalid-yolo.yaml"
    app_path = tmp_path / "app.yaml"
    app_path.write_text(yaml.safe_dump(app_values), encoding="utf-8")
    (tmp_path / "invalid-yolo.yaml").write_text(
        yaml.safe_dump({"detection": {"nano": {"identifier": "alternate.pt"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "_default_config_path", lambda: app_path)

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


def test_built_wheel_installs_top_level_yaml_defaults_without_source_leakage(
    tmp_path,
):
    project_root = Path(__file__).parents[1]
    build_root = tmp_path / "project"
    copytree(
        project_root / "src",
        build_root / "src",
        ignore=ignore_patterns("*.egg-info", "__pycache__"),
    )
    copytree(project_root / "config", build_root / "config")
    copy2(project_root / "pyproject.toml", build_root / "pyproject.toml")
    output_directory = tmp_path / "dist"
    uv_path = which("uv")
    assert uv_path is not None
    result = subprocess.run(
        [uv_path, "build", "--wheel", "--out-dir", str(output_directory)],
        cwd=build_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    wheel_path = next(output_directory.glob("uav_vision-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        app_entry = next(
            name
            for name in names
            if name.endswith(".data/data/share/uav-vision/config/app.yaml")
        )
        yolo_entry = next(
            name
            for name in names
            if name.endswith(".data/data/share/uav-vision/config/yolo.yaml")
        )
        yolo_defaults = yaml.safe_load(wheel.read(yolo_entry).decode("utf-8"))

    assert app_entry
    assert yolo_entry
    assert not any("uav_vision/config/defaults" in name for name in names)
    assert not any(name.endswith(("app.json", "yolo.json")) for name in names)
    assert_default_model_mappings(yolo_defaults)

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

    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    isolated = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                "sys.path.insert(0, sys.argv[1]); "
                "import uav_vision; "
                "assert sys.argv[1] in uav_vision.__file__; "
                "from uav_vision.config.loader import load_settings; "
                "expected = {'detection': ('models/yolo26n.pt', 640), "
                "'segmentation': ('models/yolo26n-sem.pt', 640), "
                "'depth': ('models/yolo26n-depth.pt', 768)}; "
                "settings = [load_settings(overrides={'processing': "
                "{'processing_type': mode}}) for mode in expected]; "
                "assert all((value.model.identifier, value.model.input_size) "
                "== expected[value.processing.processing_type.value] "
                "for value in settings); "
                "expected_path = (__import__('pathlib').Path(sys.argv[1]) / "
                "'share/uav-vision/config/yolo.yaml').resolve(); "
                "paths = [value.yolo.path.resolve() for value in settings]; "
                "assert all(path == expected_path for path in paths); "
                "assert all(value.yolo.origin == str(expected_path) "
                "for value in settings); "
                "assert all(sys.argv[2] not in str(path) for path in paths)"
            ),
            str(installation),
            str(project_root),
        ],
        cwd=runtime_directory,
        check=False,
        capture_output=True,
        text=True,
    )

    assert isolated.returncode == 0, isolated.stderr
