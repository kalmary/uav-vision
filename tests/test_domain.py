from datetime import datetime, timezone

import numpy as np
import pytest

from uav_vision.domain import BoundingBox, Detection, Frame, ProcessedFrame


def test_frame_accepts_a_contiguous_bgr_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    captured_at = datetime(2026, 9, 18, tzinfo=timezone.utc)

    frame = Frame(image=image, sequence=3, captured_at=captured_at)

    assert frame.width == 640
    assert frame.height == 480
    assert frame.captured_at == captured_at


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((480, 640), dtype=np.uint8),
        np.zeros((480, 640, 4), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.float32),
        np.asfortranarray(np.zeros((480, 640, 3), dtype=np.uint8)),
    ],
)
def test_frame_rejects_an_image_that_is_not_contiguous_bgr_uint8(image):
    with pytest.raises(ValueError, match="BGR"):
        Frame(image=image, sequence=0, captured_at=datetime.now(timezone.utc))


def test_frame_rejects_a_boolean_sequence_number():
    with pytest.raises(ValueError, match="sequence"):
        Frame(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            sequence=True,
            captured_at=datetime.now(timezone.utc),
        )


@pytest.mark.parametrize(
    "captured_at",
    ["2026-09-18T12:00:00Z", datetime(2026, 9, 18, 12, 0)],
)
def test_frame_rejects_an_invalid_capture_timestamp(captured_at):
    with pytest.raises(ValueError, match="timestamp"):
        Frame(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            sequence=0,
            captured_at=captured_at,
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (4.0, 2.0, 4.0, 8.0),
        (4.0, 2.0, 8.0, 2.0),
        (-1.0, 2.0, 8.0, 9.0),
    ],
)
def test_bounding_box_rejects_unordered_or_negative_coordinates(bounds):
    with pytest.raises(ValueError):
        BoundingBox(*bounds)


def test_bounding_box_rejects_boolean_coordinates():
    with pytest.raises(ValueError, match="coordinates"):
        BoundingBox(False, 2.0, 8.0, 9.0)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_detection_rejects_confidence_outside_the_probability_range(confidence):
    with pytest.raises(ValueError):
        Detection(
            class_id=0,
            class_name="person",
            confidence=confidence,
            bounding_box=BoundingBox(4.0, 2.0, 8.0, 9.0),
        )


@pytest.mark.parametrize(
    ("class_id", "class_name"),
    [(True, "person"), (0, "   ")],
)
def test_detection_rejects_invalid_class_data(class_id, class_name):
    with pytest.raises(ValueError):
        Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=0.9,
            bounding_box=BoundingBox(4.0, 2.0, 8.0, 9.0),
        )


def test_processed_frame_keeps_all_detections_for_its_frame():
    frame = Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        sequence=0,
        captured_at=datetime.now(timezone.utc),
    )
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bounding_box=BoundingBox(1.0, 1.0, 3.0, 5.0),
    )

    processed = ProcessedFrame(frame=frame, detections=(detection,))

    assert processed.frame is frame
    assert processed.detections == (detection,)
