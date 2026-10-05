import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from uav_vision.config.loader import load_settings
from uav_vision.config.models import ModelSize
from uav_vision.config.settings import (
    AppSettings,
    CaptureSettings,
    DetectionFilterSettings,
    DisplaySettings,
    LogLevel,
    LogSettings,
    ModelSettings,
    ProcessingType,
    YoloSettings,
)
from uav_vision.domain import (
    BoundingBox,
    DepthResult,
    Detection,
    DetectionResult,
    Frame,
    ProcessedFrame,
    ProcessingDiagnostics,
    SegmentationClass,
    SegmentationResult,
)
from uav_vision.output import FrameOutput
from uav_vision.output.display import DisplayOutput
from uav_vision.output.log import LogOutput, attach_secondary_failure


def resolved_settings(**values):
    return replace(load_settings(), **values)


class DisplayBackend:
    FONT_HERSHEY_SIMPLEX = 7
    COLORMAP_VIRIDIS = 16

    def __init__(self, key=-1, error=None):
        self.key = key
        self.error = error
        self.rectangle_calls = []
        self.text_calls = []
        self.resize_calls = []
        self.color_map_calls = []
        self.imshow_calls = []
        self.wait_key_calls = []
        self.destroy_calls = []

    def rectangle(self, image, start, end, color, thickness):
        self.rectangle_calls.append((image, start, end, color, thickness))

    def putText(self, image, text, origin, font, scale, color, thickness):
        self.text_calls.append((image, text, origin, font, scale, color, thickness))

    def getTextSize(self, text, font, scale, thickness):
        width = max(1, int(round(len(text) * 8 * scale))) + thickness - 1
        height = max(1, int(round(12 * scale))) + thickness - 1
        baseline = max(1, int(round(3 * scale)))
        return (width, height), baseline

    def resize(self, image, dimensions):
        self.resize_calls.append((image, dimensions))
        width, height = dimensions
        return np.full((height, width, 3), image[0, 0], dtype=image.dtype)

    def applyColorMap(self, image, color_map):
        self.color_map_calls.append((image.copy(), color_map))
        return np.stack((image, image // 2, 255 - image), axis=-1)

    def imshow(self, window_name, image):
        self.imshow_calls.append((window_name, image))
        if self.error is not None:
            raise self.error

    def waitKey(self, delay):
        self.wait_key_calls.append(delay)
        return self.key

    def destroyWindow(self, window_name):
        self.destroy_calls.append(window_name)


class TextDisplayBackend(DisplayBackend):
    def __init__(self):
        super().__init__()
        self.text_bounds = []

    def putText(self, image, text, origin, font, scale, color, thickness):
        super().putText(image, text, origin, font, scale, color, thickness)
        (width, height), baseline = self.getTextSize(text, font, scale, thickness)
        left, bottom = origin
        bounds = (left, bottom - height, left + width, bottom + baseline)
        self.text_bounds.append((image, text, bounds, scale, color, thickness))
        image_height, image_width = image.shape[:2]
        clipped_left = max(0, left)
        clipped_top = max(0, bottom - height)
        clipped_right = min(image_width, left + width)
        clipped_bottom = min(image_height, bottom + baseline)
        if clipped_left < clipped_right and clipped_top < clipped_bottom:
            image[
                clipped_top:clipped_bottom,
                clipped_left:clipped_right,
            ] = color


def frame():
    return Frame(
        image=np.full((20, 30, 3), 17, dtype=np.uint8),
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


def processed_frame(detections=(), frames_per_second=None):
    values = tuple(detections)
    return ProcessedFrame(
        frame=frame(),
        result=DetectionResult(values),
        diagnostics=ProcessingDiagnostics(
            None,
            0.0,
            None,
            len(values),
            len(values),
            frames_per_second,
        ),
    )


def segmentation_frame(
    class_map,
    classes,
    frames_per_second=None,
    diagnostics=None,
):
    return ProcessedFrame(
        frame=frame(),
        result=SegmentationResult(class_map, tuple(classes)),
        diagnostics=diagnostics
        or ProcessingDiagnostics(
            None,
            0.0,
            None,
            len(np.unique(class_map)),
            len(np.unique(class_map)),
            frames_per_second,
        ),
        processing_type=ProcessingType.SEGMENTATION,
    )


def depth_frame(depth_map, unit="metre", scale=1.0, frames_per_second=None):
    return ProcessedFrame(
        frame=frame(),
        result=DepthResult(depth_map, unit, scale),
        diagnostics=ProcessingDiagnostics(
            0.01,
            0.02,
            0.03,
            1,
            1,
            frames_per_second,
        ),
        processing_type=ProcessingType.DEPTH,
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
        "inference.model=models/yolo26n.pt "
        "inference.input_size=640 inference.device=cpu runtime.fps=30 "
        "detection.selected_classes=all detection.minimum_confidence=0.0 "
        "detection.top_k=unlimited "
        "display.enabled=false "
        "display.width=1280 display.height=720 logging.level=basic "
        "logging.destination=console "
        "yolo.origin={0}\n"
    ).format(Path(__file__).resolve().parents[1] / "config" / "yolo.yaml")

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


def test_log_startup_reports_yolo_origin_and_selected_model_pair():
    stream = StringIO()
    settings = AppSettings(
        yolo=YoloSettings(
            path=None,
            origin="packaged defaults",
            models={
                ProcessingType.DETECTION: {
                    ModelSize.NANO: ModelSettings("models/yolo26n.pt", 512)
                }
            },
        )
    )

    LogOutput(LogSettings(), stream=stream).startup(settings)

    assert "inference.model=models/yolo26n.pt" in stream.getvalue()
    assert "inference.input_size=512" in stream.getvalue()
    assert "yolo.origin=packaged defaults" in stream.getvalue()


def test_log_startup_reports_effective_detection_filters():
    stream = StringIO()
    configured = resolved_settings()
    configured = replace(
        configured,
        yolo=replace(
            configured.yolo,
            detection_filters=DetectionFilterSettings((2, 5), 0.6, 3),
        ),
    )

    LogOutput(LogSettings(), stream=stream).startup(configured)

    assert (
        "detection.selected_classes=2,5 detection.minimum_confidence=0.6 "
        "detection.top_k=3"
    ) in stream.getvalue()


def test_log_propagates_file_open_failures(tmp_path):
    path = tmp_path / "missing" / "uav.log"

    with pytest.raises(FileNotFoundError):
        LogOutput(LogSettings(path=path))


def test_log_debug_frame_record_contains_timestamp_counts_and_owned_timings():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)
    processed = ProcessedFrame(
        frame(),
        DetectionResult(()),
        ProcessingDiagnostics(0.01, 0.02, 0.03, 4, 2),
    )

    assert output.write(processed) is False

    assert stream.getvalue() == (
        "frame fps=unavailable detections=[]\n"
        "frame captured_at=2026-09-21T12:30:00+00:00 "
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
        "frame fps=unavailable "
        "detections=[class=person confidence=0.900000 "
        "bbox=(1.2,14.7,18.4,19.2), class=bus confidence=0.750000 "
        "bbox=(1.2,14.7,18.4,19.2)]\n"
    )


def test_log_basic_frame_record_reports_an_explicit_empty_detection_result():
    stream = StringIO()

    LogOutput(LogSettings(), stream=stream).write(processed_frame())

    assert stream.getvalue() == "frame fps=unavailable detections=[]\n"


def test_log_basic_frame_record_reports_measured_fps():
    stream = StringIO()

    LogOutput(LogSettings(), stream=stream).write(
        processed_frame(frames_per_second=24.5)
    )

    assert stream.getvalue() == "frame fps=24.50 detections=[]\n"


def test_log_basic_segmentation_record_counts_one_present_class_not_all_metadata():
    stream = StringIO()
    class_map = np.full((20, 30), 2, dtype=np.uint8)
    processed = segmentation_frame(
        class_map,
        (SegmentationClass(2, "sky"), SegmentationClass(7, "tree")),
    )

    LogOutput(LogSettings(), stream=stream).write(processed)

    assert stream.getvalue() == "frame fps=unavailable segmentation.classes=1\n"


def test_log_basic_segmentation_record_counts_multiple_classes_and_reports_fps():
    stream = StringIO()
    class_map = np.full((20, 30), 2, dtype=np.uint8)
    class_map[:, 15:] = 7
    processed = segmentation_frame(
        class_map,
        (SegmentationClass(2, "sky"), SegmentationClass(7, "tree")),
        frames_per_second=12.345,
    )

    LogOutput(LogSettings(), stream=stream).write(processed)

    assert stream.getvalue() == "frame fps=12.35 segmentation.classes=2\n"


def test_log_debug_segmentation_record_adds_common_frame_diagnostics():
    stream = StringIO()
    processed = segmentation_frame(
        np.full((20, 30), 2, dtype=np.uint8),
        (SegmentationClass(2, "sky"),),
        diagnostics=ProcessingDiagnostics(0.01, 0.02, 0.03, 1, 1),
    )

    LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream).write(processed)

    assert stream.getvalue() == (
        "frame fps=unavailable segmentation.classes=1\n"
        "frame captured_at=2026-09-21T12:30:00+00:00 "
        "dimensions=30x20 raw_count=1 retained_count=1 "
        "capture_duration=0.010000 inference_duration=0.020000 "
        "processing_duration=0.030000\n"
    )


def test_log_basic_depth_record_reports_cached_statistics_units_scale_and_fps():
    stream = StringIO()
    depth_map = np.full((20, 30), 2.0, dtype=np.float32)
    depth_map[:, 15:] = 4.0

    LogOutput(LogSettings(), stream=stream).write(
        depth_frame(depth_map, scale=0.001, frames_per_second=20.0)
    )

    assert stream.getvalue() == (
        "frame fps=20.00 depth.minimum=2.000000 depth.maximum=4.000000 "
        "depth.mean=3.000000 depth.standard_deviation=1.000000 "
        "depth.unit=metre depth.scale=0.001\n"
    )


def test_log_debug_depth_record_adds_raw_range_and_common_frame_diagnostics():
    stream = StringIO()
    depth_map = np.full((20, 30), 2.0, dtype=np.float32)
    depth_map[:, 15:] = 4.0

    LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream).write(
        depth_frame(depth_map)
    )

    assert stream.getvalue() == (
        "frame fps=unavailable depth.minimum=2.000000 depth.maximum=4.000000 "
        "depth.mean=3.000000 depth.standard_deviation=1.000000 "
        "depth.unit=metre depth.scale=1.0\n"
        "frame captured_at=2026-09-21T12:30:00+00:00 dimensions=30x20 "
        "raw_count=1 retained_count=1 capture_duration=0.010000 "
        "inference_duration=0.020000 processing_duration=0.030000 "
        "depth.raw_minimum=2.000000 depth.raw_maximum=4.000000\n"
    )


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


def test_log_redacts_same_line_at_signs_in_url_credentials():
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)

    output.diagnostic(
        "capture",
        "initialized source=rtsp://pilot:sec@ret@camera.local/live",
    )

    record = stream.getvalue()
    assert "rtsp://***@camera.local/live" in record
    assert "sec" not in record
    assert "ret" not in record


@pytest.mark.parametrize(
    "source",
    [
        "rtsp://pilot:sec@ret word@camera.local/live",
        "rtsp://pilot:123/secret@camera.local/live",
    ],
)
def test_log_redacts_complete_malformed_url_credentials(source):
    stream = StringIO()
    output = LogOutput(LogSettings(level=LogLevel.DEBUG), stream=stream)

    output.diagnostic("capture", "initialized source=" + source)

    assert stream.getvalue() == (
        "diagnostic component=capture event=initialized "
        "source=rtsp://***@camera.local/live\n"
    )


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
        ),
        frames_per_second=24.5,
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
        (annotated, "FPS: 24.50", (10, 20), 7, 0.5, (255, 255, 255), 1),
        (annotated, "car 0.75", (1, 5), 7, 0.5, (0, 255, 0), 1),
        (annotated, "bus 0.75", (3, 0), 7, 0.5, (0, 255, 0), 1),
    ]
    assert backend.resize_calls == [(annotated, (640, 427))]
    assert backend.imshow_calls[0][0] == "UAV Vision"
    assert backend.imshow_calls[0][1].shape == (480, 640, 3)
    assert backend.wait_key_calls == [1]


def test_display_letterboxes_wide_frames_without_distortion():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=60, height=60), backend=backend)

    output.write(processed_frame())

    displayed = backend.imshow_calls[0][1]
    assert backend.resize_calls[0][1] == (60, 40)
    assert displayed.shape == (60, 60, 3)
    assert np.all(displayed[:10] == 0)
    assert np.all(displayed[10:50] == 17)
    assert np.all(displayed[50:] == 0)


def test_display_pillarboxes_frames_for_wide_resolutions_without_distortion():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=90, height=30), backend=backend)

    output.write(processed_frame())

    displayed = backend.imshow_calls[0][1]
    assert backend.resize_calls[0][1] == (45, 30)
    assert displayed.shape == (30, 90, 3)
    assert np.all(displayed[:, :22] == 0)
    assert np.all(displayed[:, 22:67] == 17)
    assert np.all(displayed[:, 67:] == 0)


def test_display_blends_deterministic_segmentation_colours_and_orders_labels():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=30, height=20), backend=backend)
    class_map = np.full((20, 30), 2, dtype=np.uint8)
    class_map[0, 0] = 7
    processed = segmentation_frame(
        class_map,
        (SegmentationClass(7, "tree"), SegmentationClass(2, "sky")),
        frames_per_second=24.5,
    )
    original_image = processed.frame.image.copy()
    original_class_map = processed.result.class_map.copy()

    should_stop = output.write(processed)

    annotated = backend.resize_calls[0][0]
    assert should_stop is False
    assert annotated is not processed.frame.image
    assert annotated[0, 0].tolist() == [32, 96, 66]
    assert annotated[0, 1].tolist() == [61, 62, 110]
    displayed = backend.imshow_calls[0][1]
    fps_call = backend.text_calls[0]
    assert fps_call[0] is displayed
    assert fps_call[1] == "FPS: 24.50"
    assert 0 < fps_call[4] <= 0.5
    assert fps_call[-2:] == ((255, 255, 255), 1)
    label_calls = [value for value in backend.text_calls if value[-1] == 1][1:]
    assert [value[1] for value in label_calls] == ["2: sky", "7: tree"]
    assert [value[-2] for value in label_calls] == [
        (127, 131, 251),
        (56, 216, 140),
    ]
    assert np.array_equal(processed.frame.image, original_image)
    assert np.array_equal(processed.result.class_map, original_class_map)


def test_display_letterboxes_segmentation_overlay_in_configured_dimensions():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=60, height=60), backend=backend)
    processed = segmentation_frame(
        np.full((20, 30), 2, dtype=np.uint8),
        (SegmentationClass(2, "sky"),),
    )

    output.write(processed)

    displayed = backend.imshow_calls[0][1]
    assert backend.resize_calls[0][1] == (60, 40)
    assert displayed.shape == (60, 60, 3)
    assert np.all(displayed[:10] == 0)
    assert np.all(displayed[10:50] == [61, 62, 110])
    assert np.all(displayed[50:] == 0)


def test_display_places_segmentation_labels_within_the_final_canvas():
    backend = TextDisplayBackend()
    output = DisplayOutput(DisplaySettings(width=120, height=40), backend=backend)
    class_map = np.full((20, 30), 2, dtype=np.uint8)
    class_map[:, 15:] = 7
    processed = segmentation_frame(
        class_map,
        (SegmentationClass(2, "sky"), SegmentationClass(7, "tree")),
    )

    output.write(processed)

    displayed = backend.imshow_calls[0][1]
    labels = [
        value
        for value in backend.text_bounds
        if not value[1].startswith("FPS:") and value[-1] == 1
    ]
    label_draws = [
        value for value in backend.text_bounds if not value[1].startswith("FPS:")
    ]
    assert [value[1] for value in labels] == ["2: sky", "7: tree"]
    assert all(value[0] is displayed for value in labels)
    assert all(
        0 <= left < right <= displayed.shape[1]
        and 0 <= top < bottom <= displayed.shape[0]
        for _, _, (left, top, right, bottom), _, _, _ in label_draws
    )


def test_display_wraps_segmentation_labels_into_multiple_columns():
    backend = TextDisplayBackend()
    output = DisplayOutput(DisplaySettings(width=180, height=45), backend=backend)
    class_map = np.zeros((20, 30), dtype=np.uint8)
    for class_id in range(5):
        class_map[:, class_id * 6 : (class_id + 1) * 6] = class_id
    processed = segmentation_frame(
        class_map,
        tuple(
            SegmentationClass(class_id, "class-{0}".format(class_id))
            for class_id in range(5)
        ),
    )

    output.write(processed)

    labels = [
        value
        for value in backend.text_bounds
        if not value[1].startswith("FPS:") and value[-1] == 1
    ]
    left_positions = [value[2][0] for value in labels]
    top_positions = [value[2][1] for value in labels]
    assert [value[1] for value in labels] == [
        "0: class-0",
        "1: class-1",
        "2: class-2",
        "3: class-3",
        "4: class-4",
    ]
    assert len(set(left_positions)) > 1
    assert len(set(top_positions)) < len(top_positions)


def test_display_reserves_final_canvas_header_between_fps_and_segmentation_labels():
    backend = TextDisplayBackend()
    output = DisplayOutput(DisplaySettings(width=120, height=40), backend=backend)
    class_map = np.full((20, 30), 2, dtype=np.uint8)
    class_map[:, 15:] = 7
    processed = segmentation_frame(
        class_map,
        (SegmentationClass(2, "sky"), SegmentationClass(7, "tree")),
        frames_per_second=24.5,
    )

    output.write(processed)

    displayed = backend.imshow_calls[0][1]
    fps = next(
        value
        for value in backend.text_bounds
        if value[1] == "FPS: 24.50" and value[-1] == 1
    )
    labels = [
        value
        for value in backend.text_bounds
        if not value[1].startswith("FPS:") and value[-1] == 1
    ]
    label_draws = [
        value for value in backend.text_bounds if not value[1].startswith("FPS:")
    ]
    assert fps[0] is displayed
    assert fps[3] == 0.5
    assert all(value[0] is displayed for value in labels)
    fps_left, fps_top, fps_right, fps_bottom = fps[2]
    assert all(
        fps_right <= left or right <= fps_left or fps_bottom <= top or bottom <= fps_top
        for _, _, (left, top, right, bottom), _, _, _ in label_draws
    )


def test_display_scales_long_segmentation_labels_to_their_grid_cells():
    backend = TextDisplayBackend()
    output = DisplayOutput(DisplaySettings(width=160, height=40), backend=backend)
    class_map = np.zeros((20, 30), dtype=np.uint8)
    names = (
        "pedestrian-crossing-and-sidewalk",
        "multi-storey-residential-building",
        "construction-and-maintenance-vehicle",
        "vegetation-beside-the-flight-corridor",
    )
    for class_id in range(4):
        class_map[:, class_id * 7 : (class_id + 1) * 7] = class_id
    class_map[:, 28:] = 3
    processed = segmentation_frame(
        class_map,
        tuple(
            SegmentationClass(class_id, class_name)
            for class_id, class_name in enumerate(names)
        ),
    )

    output.write(processed)

    displayed = backend.imshow_calls[0][1]
    labels = [
        value
        for value in backend.text_bounds
        if not value[1].startswith("FPS:") and value[-1] == 1
    ]
    label_draws = [
        value for value in backend.text_bounds if not value[1].startswith("FPS:")
    ]
    assert [value[1] for value in labels] == [
        "{0}: {1}".format(class_id, class_name)
        for class_id, class_name in enumerate(names)
    ]
    assert all(value[3] < 0.5 for value in labels)
    assert all(
        0 <= left < right <= displayed.shape[1]
        and 0 <= top < bottom <= displayed.shape[0]
        for _, _, (left, top, right, bottom), _, _, _ in label_draws
    )
    for index, first in enumerate(labels):
        first_left, first_top, first_right, first_bottom = first[2]
        for second in labels[index + 1 :]:
            second_left, second_top, second_right, second_bottom = second[2]
            assert (
                first_right <= second_left
                or second_right <= first_left
                or first_bottom <= second_top
                or second_bottom <= first_top
            )


def test_display_normalizes_extreme_depth_values_and_applies_viridis_without_mutation():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=60, height=60), backend=backend)
    maximum = np.finfo(np.float32).max
    depth_map = np.full((20, 30), -maximum, dtype=np.float32)
    depth_map[:, 10:20] = 0.0
    depth_map[:, 20:] = maximum
    processed = depth_frame(
        depth_map,
        unit="inverse_metre",
        scale=0.001,
        frames_per_second=24.5,
    )
    original_image = processed.frame.image.copy()
    original_depth_map = processed.result.depth_map.copy()

    should_stop = output.write(processed)

    normalized, color_map = backend.color_map_calls[0]
    assert should_stop is False
    assert color_map == backend.COLORMAP_VIRIDIS
    assert normalized.dtype == np.uint8
    assert np.all(normalized[:, :10] == 0)
    assert np.all(normalized[:, 10:20] == 127)
    assert np.all(normalized[:, 20:] == 255)
    colorized = backend.resize_calls[0][0]
    assert colorized.shape == (20, 30, 3)
    assert colorized.dtype == np.uint8
    assert np.all(colorized[:, :10] == [0, 0, 255])
    assert np.all(colorized[:, 10:20] == [127, 63, 128])
    assert np.all(colorized[:, 20:] == [255, 127, 0])
    assert backend.resize_calls[0][1] == (60, 40)
    displayed = backend.imshow_calls[0][1]
    assert displayed.shape == (60, 60, 3)
    assert np.all(displayed[:10] == 0)
    assert np.all(displayed[10:50] == [0, 0, 255])
    assert np.all(displayed[50:] == 0)
    assert backend.text_calls[0][0] is displayed
    assert backend.text_calls[0][1] == "FPS: 24.50"
    assert np.array_equal(processed.frame.image, original_image)
    assert np.array_equal(processed.result.depth_map, original_depth_map)


def test_display_normalizes_the_smallest_positive_depth_span_without_overflow():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=30, height=20), backend=backend)
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    depth_map = np.zeros((20, 30), dtype=np.float64)
    depth_map[:, 15:] = smallest
    processed = depth_frame(depth_map)
    original_depth_map = processed.result.depth_map.copy()

    with np.errstate(all="raise"):
        output.write(processed)

    normalized, _ = backend.color_map_calls[0]
    assert np.all(normalized[:, :15] == 0)
    assert np.all(normalized[:, 15:] == 255)
    assert np.array_equal(processed.result.depth_map, original_depth_map)


def test_display_normalizes_a_constant_depth_map_to_zero():
    backend = DisplayBackend()
    output = DisplayOutput(DisplaySettings(width=30, height=20), backend=backend)

    output.write(depth_frame(np.full((20, 30), 7.0, dtype=np.float32)))

    normalized, color_map = backend.color_map_calls[0]
    assert color_map == backend.COLORMAP_VIRIDIS
    assert np.array_equal(normalized, np.zeros((20, 30), dtype=np.uint8))


def test_display_depth_requests_shutdown_for_quit_key():
    output = DisplayOutput(DisplaySettings(), backend=DisplayBackend(key=ord("q")))

    assert output.write(depth_frame(np.full((20, 30), 1.0, dtype=np.float32))) is True


def test_display_segmentation_requests_shutdown_for_quit_key():
    output = DisplayOutput(DisplaySettings(), backend=DisplayBackend(key=ord("q")))
    processed = segmentation_frame(
        np.full((20, 30), 2, dtype=np.uint8),
        (SegmentationClass(2, "sky"),),
    )

    assert output.write(processed) is True


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
