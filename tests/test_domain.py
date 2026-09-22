from datetime import datetime, timezone

import numpy as np
import pytest

from uav_vision.config.settings import ProcessingType
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


def diagnostics():
    return ProcessingDiagnostics(
        capture_duration=None,
        inference_duration=0.0,
        processing_duration=None,
        raw_count=0,
        retained_count=0,
    )


def test_processed_frame_keeps_the_detection_result_for_its_frame():
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

    result = DetectionResult((detection,))
    processed = ProcessedFrame(frame=frame, result=result, diagnostics=diagnostics())

    assert processed.frame is frame
    assert processed.detections == (detection,)
    assert processed.result is result


def test_detection_result_accepts_an_empty_tuple():
    assert DetectionResult(()) == DetectionResult(())


def test_detection_result_rejects_non_detection_values():
    with pytest.raises(ValueError, match="Detection result"):
        DetectionResult([])


def test_segmentation_result_owns_an_immutable_class_map_and_metadata_tuple():
    class_map = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    classes = (
        SegmentationClass(0, "background"),
        SegmentationClass(1, "person"),
    )

    result = SegmentationResult(class_map, classes)

    class_map[0, 0] = 1

    assert result.class_map is not class_map
    assert result.class_map[0, 0] == 0
    assert result.classes == classes
    with pytest.raises(AttributeError):
        result.classes += classes[:1]
    with pytest.raises(ValueError):
        result.class_map[0, 0] = 1


def test_depth_result_owns_an_immutable_depth_map():
    depth_map = np.array([[1.0, 2.0]], dtype=np.float32)

    result = DepthResult(depth_map, "metre", 0.001)
    depth_map[0, 0] = 9.0

    assert result.depth_map is not depth_map
    assert result.depth_map[0, 0] == 1.0
    with pytest.raises(ValueError):
        result.depth_map[0, 0] = 9.0


@pytest.mark.parametrize(
    ("class_map", "classes"),
    [
        (np.zeros((2, 2), dtype=np.float32), (SegmentationClass(0, "road"),)),
        (np.array([[0, 2]], dtype=np.uint8), (SegmentationClass(0, "road"),)),
        (np.array([[-1]], dtype=np.int8), (SegmentationClass(0, "road"),)),
        (np.zeros((2, 2, 1), dtype=np.uint8), (SegmentationClass(0, "road"),)),
    ],
)
def test_segmentation_result_rejects_invalid_maps_and_metadata(class_map, classes):
    with pytest.raises(ValueError):
        SegmentationResult(class_map, classes)


@pytest.mark.parametrize("class_id", [True, -1])
def test_segmentation_class_rejects_invalid_identifiers(class_id):
    with pytest.raises(ValueError, match="class"):
        SegmentationClass(class_id, "road")


@pytest.mark.parametrize(
    "depth_map",
    [
        np.array([[1.0, np.nan]], dtype=np.float32),
        np.array([[1.0, np.inf]], dtype=np.float32),
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((2, 2), dtype=np.int32),
    ],
)
def test_depth_result_rejects_invalid_maps(depth_map):
    with pytest.raises(ValueError):
        DepthResult(depth_map, "metre", 1.0)


@pytest.mark.parametrize("scale", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_depth_result_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        DepthResult(np.ones((2, 2), dtype=np.float32), "metre", scale)


@pytest.mark.parametrize(
    "processing_type,result",
    [
        (
            ProcessingType.SEGMENTATION,
            SegmentationResult(
                np.zeros((7, 8), dtype=np.uint8),
                (SegmentationClass(0, "background"),),
            ),
        ),
        (
            ProcessingType.DEPTH,
            DepthResult(np.ones((7, 8), dtype=np.float32), "metre", 1.0),
        ),
    ],
)
def test_processed_frame_rejects_result_dimensions_that_do_not_match_its_frame(
    processing_type, result
):
    frame = Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        sequence=0,
        captured_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValueError, match="dimensions"):
        ProcessedFrame(frame, result, diagnostics(), processing_type)


def test_processed_frame_accepts_each_result_variant_with_matching_dimensions():
    frame = Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        sequence=0,
        captured_at=datetime.now(timezone.utc),
    )
    segmentation = SegmentationResult(
        np.zeros((8, 8), dtype=np.uint8), (SegmentationClass(0, "background"),)
    )
    depth = DepthResult(np.ones((8, 8), dtype=np.float32), "metre", 0.001)

    assert (
        ProcessedFrame(
            frame,
            segmentation,
            diagnostics(),
            processing_type=ProcessingType.SEGMENTATION,
        ).result
        is segmentation
    )
    assert (
        ProcessedFrame(
            frame,
            depth,
            diagnostics(),
            processing_type=ProcessingType.DEPTH,
        ).result
        is depth
    )


@pytest.mark.parametrize(
    ("processing_type", "result"),
    [
        (
            ProcessingType.DETECTION,
            SegmentationResult(
                np.zeros((8, 8), dtype=np.uint8),
                (SegmentationClass(0, "background"),),
            ),
        ),
        (
            ProcessingType.DETECTION,
            DepthResult(np.ones((8, 8), dtype=np.float32), "metre", 0.001),
        ),
    ],
)
def test_processed_frame_rejects_a_result_for_a_different_processing_type(
    processing_type, result
):
    frame = Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        sequence=0,
        captured_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValueError, match="processing type"):
        ProcessedFrame(
            frame,
            result,
            diagnostics(),
            processing_type=processing_type,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        (
            DetectionResult(()),
            DepthResult(np.ones((1, 1), dtype=np.float32), "metre", 1.0),
        ),
    ],
)
def test_processed_frame_rejects_missing_or_mixed_results(value):
    frame = Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        sequence=0,
        captured_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValueError, match="result"):
        ProcessedFrame(frame, value, diagnostics())


@pytest.mark.parametrize(
    "field,value",
    [
        ("capture_duration", True),
        ("inference_duration", -0.1),
        ("inference_duration", float("nan")),
        ("processing_duration", float("inf")),
        ("raw_count", True),
        ("raw_count", -1),
        ("retained_count", 2),
    ],
)
def test_processing_diagnostics_reject_invalid_values(field, value):
    values = dict(
        capture_duration=0.1,
        inference_duration=0.2,
        processing_duration=0.3,
        raw_count=1,
        retained_count=1,
    )
    values[field] = value

    with pytest.raises(ValueError):
        ProcessingDiagnostics(**values)


def test_processing_diagnostics_are_immutable():
    value = ProcessingDiagnostics(0.1, 0.2, 0.3, 1, 1)

    with pytest.raises(AttributeError):
        value.raw_count = 2
