import sys
from datetime import datetime, timezone
from importlib import import_module

import numpy as np
import pytest

from uav_vision.config.settings import CameraType, ProcessingType
from uav_vision.domain import (
    DepthResult,
    Frame,
    SegmentationClass,
    SegmentationResult,
)


class Camera:
    def __init__(self, source, frame, events):
        self._frame = frame
        self._events = events
        self._read = False
        events.append(("camera", source))

    def read(self):
        if self._read:
            self._events.append("end stream")
            return None
        self._read = True
        self._events.append("read frame")
        return self._frame

    def close(self):
        self._events.append("close camera")


class Provider:
    def __init__(self, processing_type, events):
        self._processing_type = processing_type
        self._events = events

    def detect(self, frame):
        assert self._processing_type is ProcessingType.DETECTION
        self._events.append(("infer", frame))
        return ()

    def segment(self, frame):
        assert self._processing_type is ProcessingType.SEGMENTATION
        self._events.append(("infer", frame))
        return SegmentationResult(
            np.zeros(frame.image.shape[:2], dtype=np.uint8),
            (SegmentationClass(0, "background"),),
        )

    def estimate_depth(self, frame):
        assert self._processing_type is ProcessingType.DEPTH
        self._events.append(("infer", frame))
        return DepthResult(
            np.full(frame.image.shape[:2], 2.0, dtype=np.float32),
            "metre",
            1.0,
        )


class DisplayBackend:
    FONT_HERSHEY_SIMPLEX = 0
    COLORMAP_VIRIDIS = 1

    def __init__(self, events):
        self._events = events

    def getTextSize(self, text, font, scale, thickness):
        return (max(1, len(text) * 6), 10), 2

    def putText(self, image, *args):
        return image

    def rectangle(self, image, *args):
        return image

    def applyColorMap(self, image, color_map):
        return np.repeat(image[:, :, np.newaxis], 3, axis=2)

    def resize(self, image, dimensions):
        width, height = dimensions
        return np.zeros((height, width, 3), dtype=image.dtype)

    def imshow(self, name, image):
        self._events.append(("display frame", name, image.shape))

    def waitKey(self, delay):
        return ord("q")

    def destroyWindow(self, name):
        self._events.append("close display")


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


@pytest.mark.parametrize(
    ("entrypoint_name", "program"),
    [("main", "uav-vision"), ("usb_camera", "uav-vision-usb")],
)
@pytest.mark.parametrize("processing_type", tuple(ProcessingType))
def test_entrypoint_help_exposes_mode_specific_grouped_options(
    monkeypatch, capsys, entrypoint_name, program, processing_type
):
    from uav_vision import app

    entrypoint = import_module("uav_vision.entrypoints." + entrypoint_name)

    monkeypatch.setattr(app, "run", lambda settings: pytest.fail("app ran"))

    with pytest.raises(SystemExit):
        entrypoint.main(["--processing-type", processing_type.value, "--help"])

    help_text = capsys.readouterr().out

    assert "usage: {}".format(program) in help_text
    for group in (
        "configuration:",
        "camera:",
        "processing/inference:",
        "display:",
        "logging:",
        "runtime:",
    ):
        assert group in help_text
    for option in ("--selected-classes", "--minimum-confidence", "--top-k"):
        assert (option in help_text) is (processing_type is ProcessingType.DETECTION)
    assert "--log-level {basic,debug}" in help_text
    assert "{LogLevel.BASIC,LogLevel.DEBUG}" not in help_text
    assert "--model-path" not in help_text
    assert "--camera-source" not in help_text


@pytest.mark.parametrize(
    "entrypoint_name",
    ("main", "usb_camera"),
)
@pytest.mark.parametrize(
    ("processing_type", "model_identifier", "input_size", "result_record"),
    [
        (
            ProcessingType.DETECTION,
            "models/yolo26n.pt",
            640,
            "detections=[]",
        ),
        (
            ProcessingType.SEGMENTATION,
            "models/yolo26n-sem.pt",
            640,
            "segmentation.classes=1 segmentation.labels=[0: background color=brown]",
        ),
        (
            ProcessingType.DEPTH,
            "models/yolo26n-depth.pt",
            768,
            "depth.minimum=2.000000",
        ),
    ],
)
@pytest.mark.parametrize("display_enabled", (False, True), ids=("headless", "display"))
def test_entrypoints_run_each_mode_through_the_real_pipeline(
    monkeypatch,
    capsys,
    entrypoint_name,
    processing_type,
    model_identifier,
    input_size,
    result_record,
    display_enabled,
):
    import uav_vision.app as app

    events = []
    input_frame = Frame(
        np.zeros((6, 8, 3), dtype=np.uint8),
        datetime(2026, 10, 5, tzinfo=timezone.utc),
    )
    backend = DisplayBackend(events)
    provider = Provider(processing_type, events)
    provider_arguments = []
    selected_provider = {
        ProcessingType.DETECTION: "UltralyticsDetector",
        ProcessingType.SEGMENTATION: "UltralyticsSegmenter",
        ProcessingType.DEPTH: "UltralyticsDepthEstimator",
    }[processing_type]

    monkeypatch.setattr(
        app,
        "OpenCvCamera",
        lambda source: Camera(source, input_frame, events),
    )
    for provider_name in (
        "UltralyticsDetector",
        "UltralyticsSegmenter",
        "UltralyticsDepthEstimator",
    ):
        monkeypatch.setattr(
            app,
            provider_name,
            lambda *args, name=provider_name: pytest.fail(
                "unexpected provider {}".format(name)
            ),
        )

    def provider_factory(settings, identifier, size):
        provider_arguments.append((settings.device, identifier, size))
        return provider

    monkeypatch.setattr(app, selected_provider, provider_factory)
    monkeypatch.setitem(sys.modules, "cv2", backend)

    arguments = ["--processing-type", processing_type.value]
    if display_enabled:
        arguments.append("--display")
    entrypoint = import_module("uav_vision.entrypoints." + entrypoint_name)

    assert entrypoint.main(arguments) == 0

    log = capsys.readouterr().err
    assert provider_arguments == [("cpu", model_identifier, input_size)]
    assert events[0] == ("camera", 0)
    assert ("infer", input_frame) in events
    assert "processing.type={}".format(processing_type.value) in log
    assert "inference.model_size=nano" in log
    assert "inference.model={}".format(model_identifier) in log
    assert "inference.input_size={}".format(input_size) in log
    assert "inference.device=cpu" in log
    assert "runtime.fps=30" in log
    assert "display.enabled={}".format(str(display_enabled).lower()) in log
    assert result_record in log
    assert events[-1] == "close camera"
    if display_enabled:
        assert events[-2:] == ["close display", "close camera"]
        assert any(
            event[0] == "display frame" for event in events if isinstance(event, tuple)
        )
    else:
        assert "close display" not in events
