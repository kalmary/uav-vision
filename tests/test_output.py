import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from uav_vision.config.settings import DisplaySettings
from uav_vision.domain import BoundingBox, Detection, Frame, ProcessedFrame
from uav_vision.output import FrameOutput
from uav_vision.output.display import DisplayOutput


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
    return ProcessedFrame(frame=frame(), detections=tuple(detections))


def test_outputs_structurally_implement_frame_output():
    backend = DisplayBackend()

    assert isinstance(DisplayOutput(DisplaySettings(), backend=backend), FrameOutput)


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
