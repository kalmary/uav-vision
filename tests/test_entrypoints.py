import pytest

from uav_vision.config.settings import CameraType


def test_main_returns_zero_and_passes_parsed_settings_to_app(monkeypatch):
    from uav_vision.entrypoints import main

    settings = object()
    received = []
    monkeypatch.setattr(
        "uav_vision.config.cli.parse_args", lambda *args, **kwargs: settings
    )
    monkeypatch.setattr("uav_vision.app.run", lambda value: received.append(value))

    assert main.main(["--model-size", "small"]) == 0
    assert received == [settings]


def test_main_returns_130_for_keyboard_interrupt_after_app_run(monkeypatch):
    from uav_vision.entrypoints import main

    monkeypatch.setattr(
        "uav_vision.config.cli.parse_args", lambda *args, **kwargs: object()
    )

    def interrupt(settings):
        raise KeyboardInterrupt

    monkeypatch.setattr("uav_vision.app.run", interrupt)

    assert main.main([]) == 130


def test_main_propagates_non_keyboard_errors(monkeypatch):
    from uav_vision.entrypoints import main

    monkeypatch.setattr(
        "uav_vision.config.cli.parse_args", lambda *args, **kwargs: object()
    )

    def fail(settings):
        raise RuntimeError("failed")

    monkeypatch.setattr("uav_vision.app.run", fail)

    with pytest.raises(RuntimeError, match="failed"):
        main.main([])


def test_usb_entrypoint_uses_usb_defaults_and_keeps_cli_overrides(monkeypatch):
    from uav_vision.entrypoints import usb_camera

    values = []

    monkeypatch.setattr("uav_vision.app.run", lambda settings: values.append(settings))

    assert usb_camera.main([]) == 0
    assert values[0].capture.camera_type is CameraType.OPENCV
    assert values[0].capture.source == 0

    assert (
        usb_camera.main(
            [
                "--camera-type",
                "opencv",
                "--camera-index",
                "3",
                "--model-size",
                "small",
                "--display",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert values[1].capture.camera_type is CameraType.OPENCV
    assert values[1].capture.source == 3
    assert values[1].inference.device == "cpu"
    assert values[1].display.enabled is True


def test_entrypoint_help_does_not_run_app(monkeypatch, capsys):
    from uav_vision.entrypoints import main

    monkeypatch.setattr("uav_vision.app.run", lambda settings: pytest.fail("app ran"))

    with pytest.raises(SystemExit):
        main.main(["--help"])

    assert "usage: uav-vision" in capsys.readouterr().out
