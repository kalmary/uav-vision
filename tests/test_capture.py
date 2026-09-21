import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from uav_vision.capture import (
    CaptureInitializationError,
    CaptureReadError,
    FrameSource,
    GStreamerCamera,
    OpenCvCamera,
)


class FakeVideoCapture:
    def __init__(self, opened=True, reads=(), is_opened_error=None, read_error=None):
        self.opened = opened
        self.reads = iter(reads)
        self.is_opened_error = is_opened_error
        self.read_error = read_error
        self.release_calls = 0

    def isOpened(self):
        if self.is_opened_error is not None:
            raise self.is_opened_error
        return self.opened

    def read(self):
        if self.read_error is not None:
            raise self.read_error
        return next(self.reads, (False, None))

    def release(self):
        self.release_calls += 1


def image():
    return np.zeros((4, 6, 3), dtype=np.uint8)


def create_camera(camera_type, capture):
    if camera_type == "opencv":
        return OpenCvCamera(0, capture_factory=lambda source: capture)
    return GStreamerCamera(
        "nvarguscamerasrc ! appsink",
        capture_factory=lambda pipeline: capture,
    )


def test_open_cv_camera_structurally_implements_frame_source():
    camera = OpenCvCamera(0, capture_factory=lambda source: FakeVideoCapture())

    assert isinstance(camera, FrameSource)


@pytest.mark.parametrize("camera_type", ["opencv", "gstreamer"])
def test_camera_returns_frames_with_sequence_and_utc_timestamp(camera_type):
    capture = FakeVideoCapture(reads=((True, image()), (True, image())))
    camera = create_camera(camera_type, capture)

    first = camera.read()
    second = camera.read()

    assert first.image.shape == (4, 6, 3)
    assert first.sequence == 0
    assert second.sequence == 1
    assert first.captured_at.tzinfo is timezone.utc
    assert first.captured_at <= datetime.now(timezone.utc)


def test_open_cv_file_returns_none_at_end_of_stream():
    camera = OpenCvCamera(
        "recording.mp4",
        capture_factory=lambda source: FakeVideoCapture(),
    )

    assert camera.read() is None


@pytest.mark.parametrize("source", [0, "rtsp://camera/live"])
def test_open_cv_live_source_raises_when_a_read_fails(source):
    camera = OpenCvCamera(source, capture_factory=lambda source: FakeVideoCapture())

    with pytest.raises(CaptureReadError, match="read"):
        camera.read()


def test_gstreamer_camera_raises_when_a_read_fails():
    camera = GStreamerCamera(
        "nvarguscamerasrc ! appsink",
        capture_factory=lambda pipeline: FakeVideoCapture(),
    )

    with pytest.raises(CaptureReadError, match="read"):
        camera.read()


@pytest.mark.parametrize("camera_type", ["opencv", "gstreamer"])
def test_camera_releases_capture_when_is_opened_raises(camera_type):
    capture = FakeVideoCapture(is_opened_error=RuntimeError("backend failure"))

    with pytest.raises(CaptureInitializationError, match="open"):
        create_camera(camera_type, capture)

    assert capture.release_calls == 1


@pytest.mark.parametrize("camera_type", ["opencv", "gstreamer"])
def test_camera_converts_backend_read_exceptions_to_capture_errors(camera_type):
    capture = FakeVideoCapture(read_error=RuntimeError("backend failure"))
    camera = create_camera(camera_type, capture)

    with pytest.raises(CaptureReadError, match="read"):
        camera.read()


@pytest.mark.parametrize(
    ("camera_class", "source"),
    [
        (OpenCvCamera, 0),
        (GStreamerCamera, "nvarguscamerasrc ! appsink"),
    ],
)
def test_camera_reports_initialization_error_when_cv2_is_unavailable(
    camera_class, source, monkeypatch
):
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(CaptureInitializationError):
        camera_class(source)


@pytest.mark.parametrize("source", [True, -1, "", "   "])
def test_open_cv_camera_rejects_invalid_sources(source):
    with pytest.raises(ValueError, match="source"):
        OpenCvCamera(source, capture_factory=lambda value: FakeVideoCapture())


@pytest.mark.parametrize("pipeline", ["", "   "])
def test_gstreamer_camera_rejects_an_empty_pipeline(pipeline):
    with pytest.raises(ValueError, match="pipeline"):
        GStreamerCamera(
            pipeline,
            capture_factory=lambda value: FakeVideoCapture(),
        )


@pytest.mark.parametrize(
    ("camera_class", "arguments"),
    [
        (OpenCvCamera, (0,)),
        (GStreamerCamera, ("nvarguscamerasrc ! appsink",)),
    ],
)
def test_initialization_failure_releases_the_created_capture(camera_class, arguments):
    capture = FakeVideoCapture(opened=False)
    factory = (
        (lambda source: capture)
        if camera_class is OpenCvCamera
        else (lambda pipeline: capture)
    )

    with pytest.raises(CaptureInitializationError, match="open"):
        camera_class(*arguments, capture_factory=factory)

    assert capture.release_calls == 1


@pytest.mark.parametrize("camera_type", ["opencv", "gstreamer"])
@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((4, 6), dtype=np.uint8),
        np.zeros((4, 6, 4), dtype=np.uint8),
        np.zeros((4, 6, 3), dtype=np.float32),
        np.asfortranarray(np.zeros((4, 6, 3), dtype=np.uint8)),
        np.zeros((0, 6, 3), dtype=np.uint8),
    ],
)
def test_camera_converts_invalid_backend_images_to_capture_errors(
    camera_type, bad_image
):
    camera = create_camera(
        camera_type,
        FakeVideoCapture(reads=((True, bad_image),)),
    )

    with pytest.raises(CaptureReadError, match="image"):
        camera.read()


@pytest.mark.parametrize("source", [0, "recording.mp4", "rtsp://camera/live"])
def test_open_cv_camera_passes_its_source_to_video_capture(source):
    calls = []
    camera = OpenCvCamera(
        source,
        capture_factory=lambda value: calls.append(value) or FakeVideoCapture(),
    )

    assert calls == [source]
    camera.close()


def test_gstreamer_camera_passes_its_pipeline_and_backend_to_video_capture(monkeypatch):
    calls = []
    capture = FakeVideoCapture()
    fake_cv2 = SimpleNamespace(
        CAP_GSTREAMER=1900,
        VideoCapture=lambda pipeline, backend: (
            calls.append((pipeline, backend)) or capture
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    camera = GStreamerCamera(
        "nvarguscamerasrc ! appsink",
    )

    assert calls == [("nvarguscamerasrc ! appsink", 1900)]
    camera.close()


@pytest.mark.parametrize("camera_type", ["opencv", "gstreamer"])
def test_close_is_idempotent_and_read_after_close_fails(camera_type):
    capture = FakeVideoCapture()
    camera = create_camera(camera_type, capture)

    camera.close()
    camera.close()

    assert capture.release_calls == 1
    with pytest.raises(CaptureReadError, match="closed"):
        camera.read()


@pytest.mark.parametrize("camera_type", ["opencv", "gstreamer"])
def test_context_manager_closes_the_camera_without_suppressing_errors(camera_type):
    capture = FakeVideoCapture()
    camera = create_camera(camera_type, capture)

    with pytest.raises(RuntimeError, match="expected"):
        with camera as source:
            assert source is camera
            raise RuntimeError("expected")

    assert capture.release_calls == 1
