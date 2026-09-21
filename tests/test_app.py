import subprocess
import sys
from unittest.mock import ANY

import pytest

from uav_vision.config.settings import (
    AppSettings,
    CameraType,
    CaptureSettings,
    DisplaySettings,
)


def test_importing_app_does_not_import_display_module():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import uav_vision.app; "
            "assert 'uav_vision.output.display' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


class Resource:
    def __init__(self, name, events, close_error=None):
        self.name = name
        self.events = events
        self.close_error = close_error

    def close(self):
        self.events.append("close " + self.name)
        if self.close_error is not None:
            raise self.close_error


class LogResource(Resource):
    def startup(self, configured):
        self.events.append(("startup", configured))

    def diagnostic(self, component, event, error=None):
        self.events.append(("diagnostic", component, event, error))

    def write(self, processed):
        return False


def settings(camera_type=CameraType.OPENCV, display=None):
    source = 2
    if camera_type is CameraType.GSTREAMER:
        source = "camera ! appsink"
    return AppSettings(
        capture=CaptureSettings(camera_type, source),
        display=display,
    )


@pytest.mark.parametrize(
    ("camera_type", "camera_name"),
    [
        (CameraType.OPENCV, "opencv"),
        (CameraType.GSTREAMER, "gstreamer"),
    ],
)
def test_run_processes_headless_camera_with_logger_as_its_first_output(
    monkeypatch, camera_type, camera_name
):
    import uav_vision.app as app

    events = []
    source = Resource(camera_name, events)
    log_output = LogResource("logger", events)
    detector = object()
    processor = object()
    stop = object()

    monkeypatch.setattr(
        app,
        "OpenCvCamera",
        lambda value: events.append(("opencv", value)) or source,
    )
    monkeypatch.setattr(
        app,
        "LogOutput",
        lambda value: events.append(("logger", value)) or log_output,
    )
    monkeypatch.setattr(
        app,
        "GStreamerCamera",
        lambda value: events.append(("gstreamer", value)) or source,
    )
    monkeypatch.setattr(
        app,
        "UltralyticsDetector",
        lambda value: events.append(("detector", value)) or detector,
    )
    monkeypatch.setattr(
        app,
        "DetectionProcessor",
        lambda value: events.append(("processor", value)) or processor,
    )
    monkeypatch.setattr(
        app,
        "run_pipeline",
        lambda source_value, processor_value, outputs, should_stop: events.append(
            ("pipeline", source_value, processor_value, outputs, should_stop)
        ),
    )

    app.run(settings(camera_type), should_stop=stop)

    assert events[:-2] == [
        ("logger", settings(camera_type).logging),
        ("startup", settings(camera_type)),
        (camera_name, 2 if camera_type is CameraType.OPENCV else "camera ! appsink"),
        (
            "diagnostic",
            "capture",
            "initialized camera={0} source={1}".format(
                camera_type.value,
                2 if camera_type is CameraType.OPENCV else "camera ! appsink",
            ),
            None,
        ),
        ("detector", settings(camera_type).inference),
        (
            "diagnostic",
            "inference",
            "initialized provider=ultralytics model=nano device=default",
            None,
        ),
        ("processor", detector),
        ("diagnostic", "processing", "initialized type=detection", None),
        ("pipeline", source, processor, [log_output], stop),
    ]
    assert events[-2:] == ["close " + camera_name, "close logger"]


def test_run_uses_logger_first_and_display_as_second_output(monkeypatch):
    import uav_vision.app as app

    events = []
    source = Resource("source", events)
    log_output = LogResource("logger", events)
    display_output = Resource("display", events)
    configured = settings(display=DisplaySettings(960, 540))

    monkeypatch.setattr(app, "OpenCvCamera", lambda value: source)
    monkeypatch.setattr(app, "LogOutput", lambda value: log_output)
    monkeypatch.setattr(app, "UltralyticsDetector", lambda value: object())
    monkeypatch.setattr(app, "DetectionProcessor", lambda value: object())
    monkeypatch.setattr(
        "uav_vision.output.display.DisplayOutput",
        lambda value: events.append(("display", value)) or display_output,
    )
    monkeypatch.setattr(
        app,
        "run_pipeline",
        lambda source_value, processor, outputs, should_stop: events.append(
            ("pipeline", outputs)
        ),
    )
    app.run(configured)

    assert events == [
        ("startup", configured),
        ("diagnostic", "capture", "initialized camera=opencv source=2", None),
        (
            "diagnostic",
            "inference",
            "initialized provider=ultralytics model=nano device=default",
            None,
        ),
        ("diagnostic", "processing", "initialized type=detection", None),
        ("display", configured.display),
        ("diagnostic", "display", "initialized dimensions=960x540", None),
        ("pipeline", [log_output, display_output]),
        "close display",
        "close source",
        "close logger",
    ]


def test_headless_run_does_not_construct_display_when_cv2_is_unavailable(monkeypatch):
    import builtins

    import uav_vision.app as app

    events = []
    monkeypatch.setattr(app, "OpenCvCamera", lambda value: Resource("source", events))
    monkeypatch.setattr(app, "UltralyticsDetector", lambda value: object())
    monkeypatch.setattr(app, "DetectionProcessor", lambda value: object())
    monkeypatch.setattr(app, "run_pipeline", lambda *args: None)
    monkeypatch.setattr(
        "uav_vision.output.display.DisplayOutput",
        lambda value: pytest.fail("display constructed"),
    )
    original_import = builtins.__import__

    def import_without_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("cv2 unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cv2)

    app.run(settings())


@pytest.mark.parametrize("error", [RuntimeError("pipeline"), KeyboardInterrupt()])
def test_run_closes_all_resources_after_pipeline_failures(monkeypatch, error):
    import uav_vision.app as app

    events = []
    source = Resource("source", events)
    display_output = Resource("display", events)
    monkeypatch.setattr(app, "OpenCvCamera", lambda value: source)
    monkeypatch.setattr(app, "UltralyticsDetector", lambda value: object())
    monkeypatch.setattr(app, "DetectionProcessor", lambda value: object())
    monkeypatch.setattr(
        "uav_vision.output.display.DisplayOutput", lambda value: display_output
    )

    def fail(*args):
        raise error

    monkeypatch.setattr(app, "run_pipeline", fail)

    with pytest.raises(type(error)):
        app.run(settings(display=DisplaySettings()))

    assert events == ["close display", "close source"]


def test_run_closes_constructed_source_when_detector_initialization_fails(monkeypatch):
    import uav_vision.app as app

    events = []
    source = Resource("source", events)
    log_output = LogResource("logger", events)
    monkeypatch.setattr(app, "OpenCvCamera", lambda value: source)
    monkeypatch.setattr(app, "LogOutput", lambda value: log_output)

    def fail(value):
        raise RuntimeError("detector")

    monkeypatch.setattr(app, "UltralyticsDetector", fail)

    with pytest.raises(RuntimeError, match="detector"):
        app.run(settings())

    assert events == [
        ("startup", settings()),
        ("diagnostic", "capture", "initialized camera=opencv source=2", None),
        ("diagnostic", "inference", "initialization failed", ANY),
        "close source",
        "close logger",
    ]


def test_run_closes_prior_resources_when_display_initialization_fails(monkeypatch):
    import uav_vision.app as app

    events = []
    source = Resource("source", events)
    monkeypatch.setattr(app, "OpenCvCamera", lambda value: source)
    monkeypatch.setattr(app, "UltralyticsDetector", lambda value: object())
    monkeypatch.setattr(app, "DetectionProcessor", lambda value: object())

    def fail(value):
        raise RuntimeError("display")

    monkeypatch.setattr("uav_vision.output.display.DisplayOutput", fail)

    with pytest.raises(RuntimeError, match="display"):
        app.run(settings(display=DisplaySettings()))

    assert events == ["close source"]


def test_run_closes_source_when_display_cleanup_raises(monkeypatch):
    import uav_vision.app as app

    events = []
    source = Resource("source", events)
    display_output = Resource("display", events, RuntimeError("display close"))
    monkeypatch.setattr(app, "OpenCvCamera", lambda value: source)
    monkeypatch.setattr(app, "UltralyticsDetector", lambda value: object())
    monkeypatch.setattr(app, "DetectionProcessor", lambda value: object())
    monkeypatch.setattr(
        "uav_vision.output.display.DisplayOutput", lambda value: display_output
    )
    monkeypatch.setattr(app, "run_pipeline", lambda *args: None)

    with pytest.raises(RuntimeError, match="display close"):
        app.run(settings(display=DisplaySettings()))

    assert events == ["close display", "close source"]
