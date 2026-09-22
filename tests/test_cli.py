import pytest

from uav_vision.config.cli import parse_args
from uav_vision.config.models import ModelSize
from uav_vision.config.settings import CameraType


def test_cli_defaults_to_resolved_opencv_detection_configuration():
    settings = parse_args([])

    assert settings.capture.camera_type is CameraType.OPENCV
    assert settings.capture.source == 0
    assert settings.inference.model_size is ModelSize.NANO
    assert settings.display.enabled is False


def test_cli_accepts_an_opencv_camera_index_model_and_display_dimensions():
    settings = parse_args(
        [
            "--camera-index",
            "2",
            "--model-size",
            "small",
            "--display",
            "--display-width",
            "960",
            "--display-height",
            "540",
        ]
    )

    assert settings.capture.source == 2
    assert settings.inference.model_size is ModelSize.SMALL
    assert settings.display.enabled is True
    assert (settings.display.width, settings.display.height) == (960, 540)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--camera-type", "unsupported"],
        ["--processing-type", "unsupported"],
        ["--model-size", "unsupported"],
    ],
)
def test_cli_describes_invalid_selections(arguments, capsys):
    with pytest.raises(SystemExit):
        parse_args(arguments)

    assert "invalid" in capsys.readouterr().err


def test_cli_rejects_removed_model_path_argument(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--model-path", "detector.engine"])

    assert "unrecognized arguments" in capsys.readouterr().err


def test_cli_uses_custom_program_name_for_help(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"], prog="uav-vision-usb")

    help_text = capsys.readouterr().out
    assert "usage: uav-vision-usb" in help_text
    assert "--camera-index" in help_text
    assert "--processing-type {detection,segmentation,depth}" in help_text
