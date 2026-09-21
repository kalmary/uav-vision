import sys
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pytest

from uav_vision.config.settings import (
    AppSettings,
    CaptureSettings,
    DisplaySettings,
    LogLevel,
    LogSettings,
)
from uav_vision.domain import (
    BoundingBox,
    Detection,
    DetectionResult,
    Frame,
    ProcessedFrame,
    ProcessingDiagnostics,
)
from uav_vision.output import FrameOutput
from uav_vision.output.display import DisplayOutput
from uav_vision.output.log import LogOutput


class DisplayBackend:
    FONT_HERSHEY_SIMPLEX = 7

    def __init__(self, key=-1, error=None):
        self.key = key
        self.error = error
        self.rectangle_calls = []
        self.text_calls = []
        self.resize_calls = []
        self.imshow_calls = []
        self.wait_key_calls = []
        self.destroy_calls = []

    def rectangle(self, image, start, end, color, thickness):
        self.rectangle_calls.append((image, start, end, color, thickness))

    def putText(self, image, text, origin, font, scale, color, thickness):
        self.text_calls.append((image, text, origin, font, scale, color, thickness))

    def resize(self, image, dimensions):
        self.resize_calls.append((image, dimensions))
        return ("resized", image, dimensions)

    def imshow(self, window_name, image):
        self.imshow_calls.append((window_name, image))
        if self.error is not None:
            raise self.error

    def waitKey(self, delay):
        self.wait_key_calls.append(delay)
        return self.key

    def destroyWindow(self, window_name):
        self.destroy_calls.append(window_name)


def frame():
    return Frame(
        image=np.full((20, 30, 3), 17, dtype=np.uint8),
        sequence=8,
        captured_at=datetime(2026, 9, 21, 12, 30, tzinfo=timezone.utc),
    )


def detection(
    class_id=3,
    class_name="car",
    left=1.2,
    top=14.7,
    right=18.4,
    bottom=19.2,
):
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=0.75,
        bounding_box=BoundingBox(left, top, right, bottom),
    )


def processed_frame(detections=()):
    values = tuple(detections)
    return ProcessedFrame(
        frame=frame(),
        result=DetectionResult(values),
        diagnostics=ProcessingDiagnostics(None, 0.0, None, len(values), len(values)),
    )


def test_outputs_structurally_implement_frame_output():
    backend = DisplayBackend()

    assert isinstance(DisplayOutput(DisplaySettings(), backend=backend), FrameOutput)
    assert isinstance(LogOutput(LogSettings(), stream=StringIO()), FrameOutput)


def test_log_writes_resolved_startup_configuration_and_redacts_url_credentials():
    stream = StringIO()
    output = LogOutput(LogSettings(), stream=stream)
    configured = AppSettings(
        capture=CaptureSettings(source="rtsp://pilot:secret@camera.local/live")
    )

    output.startup(configured)

    assert stream.getvalue() == (
        "startup capture.type=opencv capture.source=rtsp://***@camera.local/live "
        "processing.type=detection inference.model_size=nano "
        "inference.model_path=None inference.device=None display=disabled "
        "logging.level=basic logging.path=None\n"
    )

    with pytest.raises(RuntimeError, match="already written"):
        output.startup(configured)


def test_log_writes_to_a_configured_file_and_closes_it_after_flushing(tmp_path):
    path = tmp_path / "uav.log"
    output = LogOutput(LogSettings(path=path))

    output.startup(AppSettings())
    output.close()

    assert path.read_text(encoding="utf-8").startswith("startup capture.type=opencv")


def test_log_propagates_file_open_failures(tmp_path):
    path = tmp_path / "missing" / "uav.log"

    with pytest.raises(FileNotFoundError):
        LogOutput(LogSettings(path=path))


def test_log_debug_frame_record_contains_frame_identity_counts_and_owned_timings():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)
    processed = ProcessedFrame(
        frame(),
        DetectionResult(()),
        ProcessingDiagnostics(0.01, 0.02, 0.03, 4, 2),
    )

    assert output.write(processed) is False

    assert stream.getvalue() == (
        "frame sequence=8 captured_at=2026-09-21T12:30:00+00:00 "
        "dimensions=30x20 raw_count=4 retained_count=2 "
        "capture_duration=0.010000 inference_duration=0.020000 "
        "processing_duration=0.030000\n"
    )


def test_log_debug_diagnostic_includes_exception_traceback():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)

    try:
        raise RuntimeError("camera unavailable")
    except RuntimeError as error:
        output.diagnostic("capture", "initialization failed", error)

    record = stream.getvalue()
    assert "diagnostic component=capture event=initialization failed\n" in record
    assert "RuntimeError: camera unavailable" in record


def test_log_debug_diagnostic_redacts_url_credentials():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)

    output.diagnostic(
        "capture",
        "initialized source=rtsp://pilot:secret@camera.local/live",
    )

    assert "rtsp://***@camera.local/live" in stream.getvalue()
    assert "secret" not in stream.getvalue()


def test_log_close_flushes_but_does_not_close_a_borrowed_stream():
    class Stream:
        def __init__(self):
            self.closed = False
            self.flush_calls = 0

        def write(self, value):
            return len(value)

        def flush(self):
            self.flush_calls += 1

        def close(self):
            self.closed = True

    stream = Stream()
    output = LogOutput(LogSettings(), stream=stream)
    output.startup(AppSettings())

    output.close()
    output.close()

    assert stream.flush_calls == 2
    assert stream.closed is False


def test_display_annotates_a_copy_and_shows_the_requested_dimensions():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=640, height=480), backend=backend)
    source = processed_frame(
        (
            detection(),
            detection(
                class_id=5,
                class_name="bus",
                left=3.4,
                top=2.2,
                right=15.8,
                bottom=12.6,
            ),
        )
    )
    original = source.frame.image.copy()

    should_stop = output.write(source)

    annotated = backend.rectangle_calls[0][0]
    assert should_stop is False
    assert annotated is not source.frame.image
    assert np.array_equal(source.frame.image, original)
    assert backend.rectangle_calls == [
        (annotated, (1, 15), (18, 19), (0, 255, 0), 2),
        (annotated, (3, 2), (16, 13), (0, 255, 0), 2),
    ]
    assert backend.text_calls == [
        (annotated, "car", (1, 5), 7, 0.5, (0, 255, 0), 1),
        (annotated, "bus", (3, 0), 7, 0.5, (0, 255, 0), 1),
    ]
    assert backend.resize_calls == [(annotated, (640, 480))]
    assert backend.imshow_calls == [("UAV Vision", ("resized", annotated, (640, 480)))]
    assert backend.wait_key_calls == [1]


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_display_requests_shutdown_for_quit_keys(key):
    output = DisplayOutput(DisplaySettings(), backend=DisplayBackend(key=key))

    assert output.write(processed_frame()) is True


@pytest.mark.parametrize("key", [-1, ord("x"), 0x100 + ord("x")])
def test_display_continues_for_other_keys(key):
    output = DisplayOutput(DisplaySettings(), backend=DisplayBackend(key=key))

    assert output.write(processed_frame()) is False


def test_display_rejects_writes_after_close():
    output = DisplayOutput(DisplaySettings(), backend=DisplayBackend())
    output.close()

    with pytest.raises(RuntimeError, match="closed"):
        output.write(processed_frame())


def test_display_close_destroys_its_opened_window_once():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(), backend=backend, window_name="camera")
    output.write(processed_frame())

    output.close()
    output.close()

    assert backend.destroy_calls == ["camera"]


def test_display_close_after_backend_error_destroys_its_window():
    backend = DisplayBackend(error=RuntimeError("display failed"))
    output = DisplayOutput(DisplaySettings(), backend=backend)

    with pytest.raises(RuntimeError, match="display failed"):
        output.write(processed_frame())
    output.close()

    assert backend.destroy_calls == ["UAV Vision"]


def test_display_raises_runtime_error_with_cause_when_cv2_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(RuntimeError, match="OpenCV") as raised:
        DisplayOutput(DisplaySettings())

    assert isinstance(raised.value.__cause__, ImportError)
