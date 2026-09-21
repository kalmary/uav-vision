import pytest

from uav_vision.config.cli import parse_args
from uav_vision.config.models import ModelSize
from uav_vision.config.settings import CameraType, CaptureSettings


def test_cli_defaults_to_headless_opencv_detection_with_camera_zero():
    settings = parse_args([])

    assert settings.capture.camera_type is CameraType.OPENCV
    assert settings.capture.source == 0
    assert settings.inference.model_size is ModelSize.NANO
    assert settings.display is None


def test_cli_accepts_a_gstreamer_source_explicit_model_and_display_dimensions():
    settings = parse_args(
        [
            "--camera-type",
            "gstreamer",
            "--camera-source",
            "nvarguscamerasrc ! appsink",
            "--model-path",
            "detector.engine",
            "--device",
            "cuda:0",
            "--display",
            "--display-width",
            "960",
            "--display-height",
            "540",
        ]
    )

    assert settings.capture.camera_type is CameraType.GSTREAMER
    assert settings.capture.source == "nvarguscamerasrc ! appsink"
    assert str(settings.inference.model_path) == "detector.engine"
    assert settings.inference.model_size is None
    assert settings.inference.device == "cuda:0"
    assert settings.display is not None
    assert settings.display.width == 960
    assert settings.display.height == 540


def test_cli_accepts_a_numeric_camera_source_and_model_size():
    settings = parse_args(["--camera-source", "2", "--model-size", "small"])

    assert settings.capture.source == 2
    assert settings.inference.model_size is ModelSize.SMALL


def test_cli_rejects_conflicting_model_arguments(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--model-size", "small", "--model-path", "detector.engine"])

    error = capsys.readouterr().err
    assert "not allowed with argument" in error
    assert "--model-path" in error


@pytest.mark.parametrize(
    "arguments",
    [
        ["--camera-type", "unsupported"],
        ["--processing-type", "unsupported"],
        ["--model-size", "unsupported"],
    ],
)
def test_cli_describes_an_invalid_selection(arguments, capsys):
    with pytest.raises(SystemExit):
        parse_args(arguments)

    error = capsys.readouterr().err
    assert arguments[0] in error
    assert "invalid" in error


def test_cli_rejects_an_empty_model_path(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--model-path", ""])

    assert "model path" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--display-width", "0"], "positive"),
        (["--display-height", "-1"], "positive"),
        (["--camera-source", ""], "camera source"),
        (["--device", ""], "device"),
    ],
)
def test_cli_rejects_invalid_argument_values(arguments, message, capsys):
    with pytest.raises(SystemExit):
        parse_args(arguments)

    assert message in capsys.readouterr().err


def test_cli_uses_injected_capture_defaults_and_allows_overrides():
    settings = parse_args(
        [],
        capture_defaults=CaptureSettings(CameraType.GSTREAMER, "camera ! appsink"),
    )

    assert settings.capture == CaptureSettings(CameraType.GSTREAMER, "camera ! appsink")

    settings = parse_args(
        ["--camera-type", "opencv", "--camera-source", "3"],
        capture_defaults=CaptureSettings(CameraType.GSTREAMER, "camera ! appsink"),
    )

    assert settings.capture == CaptureSettings(CameraType.OPENCV, 3)


def test_cli_uses_custom_program_name_for_help(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"], prog="uav-vision-usb")

    help_text = capsys.readouterr().out
    assert "usage: uav-vision-usb" in help_text
    assert "--camera-type {opencv,gstreamer}" in help_text
    assert "--processing-type {detection}" in help_text
    assert "--model-size {nano,small,medium,large,xlarge}" in help_text
