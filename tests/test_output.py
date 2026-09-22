import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pytest

from uav_vision.config.loader import load_settings
from uav_vision.config.models import ModelSize
from uav_vision.config.settings import (
    AppSettings,
    CaptureSettings,
    DisplaySettings,
    LogLevel,
    LogSettings,
    ProcessingType,
    YoloSettings,
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
from uav_vision.output.log import LogOutput, attach_secondary_failure


def resolved_settings(**values):
    return replace(load_settings(), **values)


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
    confidence=0.75,
    left=1.2,
    top=14.7,
    right=18.4,
    bottom=19.2,
):
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
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
    configured = resolved_settings(
        capture=CaptureSettings(source="rtsp://pilot:secret@camera.local/live")
    )

    output.startup(configured)

    assert stream.getvalue() == (
        "startup camera.type=opencv camera.source=rtsp://***@camera.local/live "
        "processing.type=detection inference.model_size=nano "
        "inference.model=yolo26n.pt "
        "inference.device=cpu runtime.fps=30 display.enabled=false "
        "display.width=1280 display.height=720 logging.level=basic "
        "logging.destination=console "
        "yolo.origin=package:uav_vision.config.defaults/yolo.json\n"
    )

    with pytest.raises(RuntimeError, match="already written"):
        output.startup(configured)


def test_log_startup_rejects_unresolved_settings_without_default_loading():
    output = LogOutput(LogSettings(), stream=StringIO())

    with pytest.raises(ValueError, match="YOLO configuration"):
        output.startup(AppSettings())


def test_log_writes_to_a_configured_file_and_closes_it_after_flushing(tmp_path):
    path = tmp_path / "uav.log"
    output = LogOutput(LogSettings(path=path))

    output.startup(resolved_settings())
    output.close()

    assert path.read_text(encoding="utf-8").startswith("startup camera.type=opencv")


def test_log_startup_reports_yolo_origin_and_selected_model():
    stream = StringIO()
    settings = AppSettings(
        yolo=YoloSettings(
            path=None,
            origin="packaged defaults",
            models={ProcessingType.DETECTION: {ModelSize.NANO: "yolo26n.pt"}},
        )
    )

    LogOutput(LogSettings(), stream=stream).startup(settings)

    assert "inference.model=yolo26n.pt" in stream.getvalue()
    assert "yolo.origin=packaged defaults" in stream.getvalue()


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
        "frame sequence=8 detections=[]\n"
        "frame sequence=8 captured_at=2026-09-21T12:30:00+00:00 "
        "dimensions=30x20 raw_count=4 retained_count=2 "
        "capture_duration=0.010000 inference_duration=0.020000 "
        "processing_duration=0.030000\n"
    )


def test_log_basic_frame_record_reports_each_retained_detection_in_order():
    stream = StringIO()
    output = LogOutput(LogSettings(), stream=stream)
    first = detection(class_name="person", confidence=0.9)
    second = detection(class_id=5, class_name="bus", confidence=0.75)

    output.write(processed_frame((first, second)))

    assert stream.getvalue() == (
        "frame sequence=8 detections=[class=person confidence=0.900000 "
        "bbox=(1.2,14.7,18.4,19.2), class=bus confidence=0.750000 "
        "bbox=(1.2,14.7,18.4,19.2)]\n"
    )


def test_log_basic_frame_record_reports_an_explicit_empty_detection_result():
    stream = StringIO()

    LogOutput(LogSettings(), stream=stream).write(processed_frame())

    assert stream.getvalue() == "frame sequence=8 detections=[]\n"


def test_log_debug_diagnostic_includes_exception_traceback():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)

    try:
        raise RuntimeError("camera unavailable")
    except RuntimeError as error:
        output.diagnostic("capture", "initialization failed", error)

    record = stream.getvalue()
    assert (
        "diagnostic component=capture event=initialization failed "
        "error=RuntimeError: camera unavailable\n"
    ) in record
    assert "RuntimeError: camera unavailable" in record


def test_log_basic_diagnostic_reports_a_concise_error_without_traceback():
    stream = StringIO()
    output = LogOutput(LogSettings(), stream=stream)

    output.diagnostic("capture", "initialization failed", RuntimeError("offline"))

    assert stream.getvalue() == (
        "diagnostic component=capture event=initialization failed "
        "error=RuntimeError: offline\n"
    )


def test_log_redacts_credentials_from_malformed_source_and_debug_traceback():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)
    settings = resolved_settings(
        capture=CaptureSettings(
            source="rtsp://pilot secret:pass/word@camera.local/live"
        )
    )

    output.startup(settings)
    try:
        raise RuntimeError("camera rtsp://pilot secret:pass/word@camera.local/live")
    except RuntimeError as error:
        output.diagnostic("capture", "initialization failed", error)

    record = stream.getvalue()
    assert record.count("rtsp://***@camera.local/live") >= 2
    assert "pilot secret" not in record
    assert "pass/word" not in record


def test_log_redacts_each_uri_segment_with_newline_credentials():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)
    first = "rtsp://first user:one@two\nthree@first.example/live"
    second = "http://second user:three@second.example/status"

    output.diagnostic("capture", "sources {0} and {1}".format(first, second))

    record = stream.getvalue()
    assert "rtsp://***@first.example/live" in record
    assert "http://***@second.example/status" in record
    assert "first user" not in record
    assert "one@two\nthree" not in record
    assert "second user" not in record
    assert "three" not in record


def test_log_preserves_uri_without_userinfo_and_an_unrelated_email_address():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)
    source = "rtsp://camera.local:554/live operator@example.com"

    output.diagnostic("capture", "initialized source=" + source)

    assert stream.getvalue() == (
        "diagnostic component=capture event=initialized source=" + source + "\n"
    )


def test_log_redacts_camera_credentials_in_query_and_gstreamer_properties():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)
    source = (
        "rtspsrc location=rtsp://camera.local/live?user-id=pilot&user-pw=secret "
        "password=another token=access keep=visible"
    )
    configured = resolved_settings(capture=CaptureSettings(source=source))

    output.startup(configured)
    try:
        raise RuntimeError("camera source: " + source)
    except RuntimeError as error:
        output.diagnostic("capture", "initialization failed", error)

    record = stream.getvalue()
    assert "user-id=***" in record
    assert "user-pw=***" in record
    assert "password=***" in record
    assert "token=***" in record
    assert "pilot" not in record
    assert "secret" not in record
    assert "another" not in record
    assert "access" not in record
    assert "keep=visible" in record


def test_log_writes_the_same_record_to_stderr_and_file_destinations(
    monkeypatch, tmp_path
):
    console = StringIO()
    monkeypatch.setattr(sys, "stderr", console)
    settings = resolved_settings(capture=CaptureSettings(source="camera ?token=secret"))
    path = tmp_path / "uav.log"

    console_output = LogOutput(LogSettings())
    console_output.startup(settings)
    console_output.diagnostic(
        "capture", "initialization failed", RuntimeError("offline")
    )
    file_output = LogOutput(LogSettings(path=path))
    file_output.startup(settings)
    file_output.diagnostic("capture", "initialization failed", RuntimeError("offline"))
    file_output.close()

    assert (
        console.getvalue().splitlines()[-1]
        == path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert "token=***" in console.getvalue()


def test_log_propagates_stderr_write_failure(monkeypatch):
    class Stream:
        def write(self, value):
            raise OSError("stderr unavailable")

        def flush(self):
            return None

    monkeypatch.setattr(sys, "stderr", Stream())

    with pytest.raises(OSError, match="stderr unavailable"):
        LogOutput(LogSettings()).startup(resolved_settings())


def test_log_propagates_write_and_flush_errors_without_a_primary_exception():
    class Stream:
        def __init__(self, write_error=None, flush_error=None):
            self._write_error = write_error
            self._flush_error = flush_error

        def write(self, value):
            if self._write_error is not None:
                raise self._write_error
            return len(value)

        def flush(self):
            if self._flush_error is not None:
                raise self._flush_error

    write_error = OSError("write failed")
    with pytest.raises(OSError) as raised:
        LogOutput(LogSettings(), stream=Stream(write_error=write_error)).startup(
            resolved_settings()
        )
    assert raised.value is write_error

    flush_error = OSError("flush failed")
    with pytest.raises(OSError) as raised:
        LogOutput(LogSettings(), stream=Stream(flush_error=flush_error)).startup(
            resolved_settings()
        )
    assert raised.value is flush_error


def test_log_close_preserves_active_exception_and_chains_flush_failure():
    class Stream:
        def write(self, value):
            return len(value)

        def flush(self):
            raise OSError("flush failed")

    output = LogOutput(LogSettings(), stream=Stream())
    primary = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt) as raised:
        with output:
            raise primary

    assert raised.value is primary
    assert isinstance(primary.__context__, OSError)
    assert str(primary.__context__) == "flush failed"


def test_log_close_appends_flush_failure_after_existing_exception_context():
    class Stream:
        def write(self, value):
            return len(value)

        def flush(self):
            raise OSError("flush failed")

    output = LogOutput(LogSettings(), stream=Stream())
    primary = RuntimeError("primary")
    previous = ValueError("previous")

    try:
        raise previous
    except ValueError:
        with pytest.raises(RuntimeError):
            try:
                raise primary
            except RuntimeError:
                with output:
                    raise

    assert isinstance(primary.__context__, OSError)
    assert str(primary.__context__) == "flush failed"
    assert primary.__context__.__context__ is previous


def test_secondary_failure_chain_is_acyclic_after_write_diagnostic_and_close_failures():
    primary = KeyboardInterrupt("write failed")
    diagnostic_error = OSError("diagnostic failed")
    previous = ValueError("previous failure")
    close_error = OSError("close failed")
    diagnostic_error.__context__ = primary
    primary.__context__ = previous

    attach_secondary_failure(primary, diagnostic_error)
    attach_secondary_failure(primary, close_error)

    assert primary.__context__ is close_error
    assert close_error.__context__ is diagnostic_error
    assert diagnostic_error.__context__ is previous
    formatted = "".join(
        traceback.format_exception(type(primary), primary, primary.__traceback__)
    )
    assert "KeyboardInterrupt: write failed" in formatted
    assert "OSError: diagnostic failed" in formatted
    assert "OSError: close failed" in formatted


def test_log_write_diagnostic_and_close_failures_keep_write_failure_primary():
    class Stream:
        def __init__(self):
            self._write_errors = [OSError("write failed"), OSError("diagnostic failed")]

        def write(self, value):
            raise self._write_errors.pop(0)

        def flush(self):
            raise OSError("close failed")

    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=Stream())
    processed = ProcessedFrame(
        frame(),
        DetectionResult(()),
        ProcessingDiagnostics(0.0, 0.0, 0.0, 0, 0),
    )
    with pytest.raises(OSError) as raised:
        with output:
            try:
                output.write(processed)
            except OSError as primary:
                try:
                    output.diagnostic("pipeline", "failed", primary)
                except OSError as diagnostic_error:
                    attach_secondary_failure(primary, diagnostic_error)
                raise

    assert str(raised.value) == "write failed"
    assert str(raised.value.__context__) == "close failed"
    assert str(raised.value.__context__.__context__) == "diagnostic failed"
    formatted = "".join(
        traceback.format_exception(
            type(raised.value),
            raised.value,
            raised.value.__traceback__,
        )
    )
    assert "OSError: write failed" in formatted
    assert "OSError: diagnostic failed" in formatted
    assert "OSError: close failed" in formatted


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
    output.startup(resolved_settings())

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
        (annotated, "car 0.75", (1, 5), 7, 0.5, (0, 255, 0), 1),
        (annotated, "bus 0.75", (3, 0), 7, 0.5, (0, 255, 0), 1),
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
