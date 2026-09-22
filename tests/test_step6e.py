import json
from io import StringIO

import pytest

from uav_vision.config.cli import parse_args
from uav_vision.config import loader
from uav_vision.config.loader import load_settings
from uav_vision.config.models import ModelSize
from uav_vision.config.settings import CameraType, ProcessingType
from uav_vision.output.log import LogOutput


def test_cli_loads_packaged_defaults_and_reports_resolved_startup_values():
    settings = parse_args([])

    assert settings.capture.camera_type is CameraType.OPENCV
    assert settings.capture.source == 0
    assert settings.inference.model_size is ModelSize.NANO
    assert settings.inference.device == "cpu"
    assert settings.fps == 30
    assert settings.display.enabled is False
    assert (settings.display.width, settings.display.height) == (1280, 720)
    assert settings.model_identifier == "yolo26n.pt"

    stream = StringIO()
    LogOutput(settings.logging, stream=stream).startup(settings)

    assert stream.getvalue() == (
        "startup camera.type=opencv camera.index=0 processing.type=detection "
        "inference.model_size=nano inference.model=yolo26n.pt "
        "inference.device=cpu runtime.fps=30 display.enabled=false "
        "display.width=1280 display.height=720 logging.level=basic "
        "logging.destination=console "
        "yolo.origin=package:uav_vision.config.defaults/yolo.json\n"
    )


def test_cli_uses_only_explicit_arguments_as_configuration_overrides(tmp_path):
    path = tmp_path / "vision.json"
    path.write_text(
        json.dumps(
            {
                "capture": {"camera_type": "gstreamer", "source": "camera ! appsink"},
                "inference": {"model_size": "small"},
                "display": {"enabled": True, "width": 960, "height": 540},
                "log_level": "debug",
                "fps": 24,
            }
        ),
        encoding="utf-8",
    )

    settings = parse_args(
        ["--config-path", str(path), "--model-size", "medium", "--no-display"]
    )

    assert settings.capture.source == "camera ! appsink"
    assert settings.inference.model_size is ModelSize.MEDIUM
    assert settings.display.enabled is False
    assert settings.display.width == 960
    assert settings.logging.level.value == "debug"
    assert settings.fps == 24


def test_cli_groups_detection_options_and_rejects_unavailable_mode(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"])

    help_text = capsys.readouterr().out
    for heading in (
        "configuration:",
        "camera:",
        "processing/inference:",
        "display:",
        "logging:",
        "runtime:",
    ):
        assert heading in help_text
    assert "--selected-classes" in help_text
    assert "--display-width" in help_text

    with pytest.raises(SystemExit):
        parse_args(["--processing-type", "segmentation"])

    assert "no model or processor is configured" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--camera-index", "-1"], "non-negative"),
        (["--camera-index", "camera"], "non-negative"),
        (["--fps", "0"], "positive"),
        (["--device", "cuda:0"], "invalid choice"),
        (["--camera-type", "gstreamer", "--camera-index", "1"], "pipeline"),
    ],
)
def test_cli_rejects_invalid_runtime_values(arguments, message, capsys):
    with pytest.raises(SystemExit):
        parse_args(arguments)

    assert message in capsys.readouterr().err


def test_loader_rejects_requested_cuda_when_the_injected_check_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr("uav_vision.config.loader.cuda_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        load_settings(overrides={"inference": {"device": "cuda"}})


def test_cli_device_override_is_applied_before_cuda_availability_check(
    tmp_path, monkeypatch
):
    path = tmp_path / "vision.json"
    path.write_text(json.dumps({"inference": {"device": "cuda"}}), encoding="utf-8")
    monkeypatch.setattr("uav_vision.config.loader.cuda_available", lambda: False)

    settings = parse_args(["--config-path", str(path), "--device", "cpu"])

    assert settings.inference.device == "cpu"


def test_cli_detection_filter_arguments_override_yolo_configuration(tmp_path):
    yolo = tmp_path / "yolo.json"
    yolo.write_text(
        json.dumps(
            {
                "detection_filters": {
                    "selected_classes": [1],
                    "minimum_confidence": 0.5,
                    "top_k": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    app = tmp_path / "app.json"
    app.write_text(json.dumps({"yolo_config_path": "yolo.json"}), encoding="utf-8")

    settings = parse_args(
        [
            "--config-path",
            str(app),
            "--selected-classes",
            "3",
            "4",
            "--minimum-confidence",
            "0.7",
            "--top-k",
            "5",
        ]
    )

    assert settings.processing.processing_type is ProcessingType.DETECTION
    assert settings.yolo.detection_filters.selected_classes == (3, 4)
    assert settings.yolo.detection_filters.minimum_confidence == 0.7
    assert settings.yolo.detection_filters.top_k == 5


@pytest.mark.parametrize("processing_type", ("segmentation", "depth"))
def test_mode_help_excludes_detection_filter_arguments(processing_type, capsys):
    with pytest.raises(SystemExit):
        parse_args(["--processing-type", processing_type, "--help"])

    help_text = capsys.readouterr().out
    assert "processing/inference:" in help_text
    assert "--selected-classes" not in help_text
    assert "--minimum-confidence" not in help_text
    assert "--top-k" not in help_text


@pytest.mark.parametrize("model_size", list(ModelSize))
def test_every_model_size_resolves_an_identifier(model_size):
    settings = parse_args(["--model-size", model_size.value])

    assert (
        settings.model_identifier
        == settings.yolo.models[ProcessingType.DETECTION][model_size]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--fps", "-1"],
        ["--fps", "fps"],
        ["--fps", "1.5"],
        ["--display-width", "0"],
        ["--display-width", "-1"],
        ["--display-width", "wide"],
        ["--display-height", "0"],
        ["--display-height", "-1"],
        ["--display-height", "tall"],
    ],
)
def test_cli_rejects_invalid_fps_and_display_dimensions(arguments, capsys):
    with pytest.raises(SystemExit):
        parse_args(arguments)

    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("contents", ("{", '{"unknown": true}'))
def test_cli_reports_bootstrap_configuration_failures_as_parser_errors(
    tmp_path, contents, capsys
):
    path = tmp_path / "vision.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit):
        parse_args(["--config-path", str(path)])

    error = capsys.readouterr().err
    assert error.startswith("usage: uav-vision")
    assert "uav-vision: error:" in error


def test_cli_reports_a_missing_bootstrap_configuration_as_a_parser_error(
    tmp_path, capsys
):
    with pytest.raises(SystemExit):
        parse_args(["--config-path", str(tmp_path / "missing.json")])

    assert (
        "uav-vision: error: Configuration file does not exist"
        in capsys.readouterr().err
    )


def test_mode_help_uses_the_processing_type_from_selected_configuration(
    tmp_path, capsys
):
    path = tmp_path / "vision.json"
    path.write_text(
        json.dumps({"processing": {"processing_type": "segmentation"}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as raised:
        parse_args(["--config-path", str(path), "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: uav-vision" in help_text
    assert "--selected-classes" not in help_text


def test_loader_rejects_an_unconfigured_processing_model_combination():
    with pytest.raises(ValueError, match="does not support"):
        load_settings(overrides={"processing": {"processing_type": "segmentation"}})


@pytest.mark.parametrize("processing_type", ("segmentation", "depth"))
def test_cli_rejects_execution_of_unimplemented_processing_modes(
    processing_type, capsys
):
    with pytest.raises(SystemExit):
        parse_args(["--processing-type", processing_type])

    assert "no model or processor is configured" in capsys.readouterr().err


def test_startup_uses_custom_configuration_and_cli_override_values(tmp_path):
    path = tmp_path / "vision.json"
    path.write_text(
        json.dumps(
            {
                "capture": {"source": 2},
                "inference": {"model_size": "small"},
                "display": {"enabled": True, "width": 960, "height": 540},
                "log_level": "debug",
                "fps": 24,
            }
        ),
        encoding="utf-8",
    )
    settings = parse_args(
        ["--config-path", str(path), "--camera-index", "4", "--fps", "20"]
    )
    stream = StringIO()

    LogOutput(settings.logging, stream=stream).startup(settings)

    assert stream.getvalue() == (
        "startup camera.type=opencv camera.index=4 processing.type=detection "
        "inference.model_size=small inference.model=yolo26s.pt "
        "inference.device=cpu runtime.fps=20 display.enabled=true "
        "display.width=960 display.height=540 logging.level=debug "
        "logging.destination=console "
        "yolo.origin=package:uav_vision.config.defaults/yolo.json\n"
    )


def test_cli_reads_selected_configuration_once_and_uses_the_same_base_for_mode(
    tmp_path, monkeypatch
):
    path = tmp_path / "vision.json"
    path.write_text(
        json.dumps({"processing": {"processing_type": "segmentation"}}),
        encoding="utf-8",
    )
    read_json = loader._read_json
    reads = []

    def counted_read_json(value):
        reads.append(value)
        return read_json(value)

    monkeypatch.setattr(loader, "_read_json", counted_read_json)

    settings = parse_args(
        [
            "--config-path",
            str(path),
            "--processing-type",
            "detection",
            "--selected-classes",
            "3",
        ]
    )

    assert reads == [path]
    assert settings.processing.processing_type is ProcessingType.DETECTION
    assert settings.yolo.detection_filters.selected_classes == (3,)


def test_cli_logging_overrides_win_and_startup_reports_file_destination(tmp_path):
    config_path = tmp_path / "vision.json"
    user_log_path = tmp_path / "user.log"
    cli_log_path = tmp_path / "cli.log"
    config_path.write_text(
        json.dumps({"log_level": "debug", "log_path": str(user_log_path)}),
        encoding="utf-8",
    )

    settings = parse_args(
        [
            "--config-path",
            str(config_path),
            "--log-level",
            "basic",
            "--log-path",
            str(cli_log_path),
        ]
    )
    output = LogOutput(settings.logging)
    output.startup(settings)
    output.close()

    assert settings.logging.level.value == "basic"
    assert settings.logging.path == cli_log_path
    assert not user_log_path.exists()
    assert "logging.level=basic" in cli_log_path.read_text(encoding="utf-8")
    assert "logging.destination={}".format(cli_log_path) in cli_log_path.read_text(
        encoding="utf-8"
    )
