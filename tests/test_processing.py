from datetime import datetime, timezone

import numpy as np
import pytest

from uav_vision.config.settings import DetectionFilterSettings
from uav_vision.domain import BoundingBox, Detection, Frame
from uav_vision.processing import DetectionProcessor, FrameProcessor


class DetectorDouble:
    def __init__(self, detections=(), error=None):
        self.detections = detections
        self.error = error
        self.calls = []

    def detect(self, input_frame):
        self.calls.append(input_frame)
        if self.error is not None:
            raise self.error
        return self.detections


def frame():
    return Frame(
        image=np.zeros((12, 16, 3), dtype=np.uint8),
        sequence=4,
        captured_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )


def detection(class_id=3, confidence=0.75):
    return Detection(
        class_id=class_id,
        class_name="class-{}".format(class_id),
        confidence=confidence,
        bounding_box=BoundingBox(1.0, 2.0, 8.0, 10.0),
    )


def test_detection_processor_structurally_implements_frame_processor():
    processor = DetectionProcessor(DetectorDouble())

    assert isinstance(processor, FrameProcessor)


def test_detection_processor_returns_detector_results_for_the_same_frame():
    expected = (detection(),)
    detector = DetectorDouble(expected)
    processor = DetectionProcessor(detector)
    input_frame = frame()

    result = processor.process(input_frame)

    assert detector.calls == [input_frame]
    assert result.frame is input_frame
    assert result.detections == expected
    assert result.diagnostics.raw_count == 1
    assert result.diagnostics.retained_count == 1


def test_detection_processor_preserves_empty_detection_results():
    detector = DetectorDouble()
    processor = DetectionProcessor(detector)

    result = processor.process(frame())

    assert result.detections == ()
    assert len(detector.calls) == 1


def test_detection_processor_orders_unfiltered_detections_by_descending_confidence():
    first = detection(class_id=1, confidence=0.4)
    second = detection(class_id=2, confidence=0.9)
    third = detection(class_id=3, confidence=0.9)
    processor = DetectionProcessor(DetectorDouble((first, second, third)))

    result = processor.process(frame())

    assert result.detections == (second, third, first)
    assert result.diagnostics.raw_count == 3
    assert result.diagnostics.retained_count == 3


@pytest.mark.parametrize(
    ("selected_classes", "expected"),
    [
        ((3,), ("first",)),
        ((2, 3, 5), ("first", "second")),
    ],
)
def test_detection_processor_filters_classes_before_confidence_and_preserves_ties(
    selected_classes, expected
):
    rejected_class = detection(class_id=1, confidence=0.99)
    below_threshold = detection(class_id=2, confidence=0.74)
    first_tie = detection(class_id=3, confidence=0.75)
    second_tie = detection(class_id=5, confidence=0.75)
    processor = DetectionProcessor(
        DetectorDouble((rejected_class, below_threshold, first_tie, second_tie)),
        DetectionFilterSettings(selected_classes, 0.75),
    )

    result = processor.process(frame())

    detections = {"first": first_tie, "second": second_tie}
    assert result.detections == tuple(detections[name] for name in expected)
    assert result.diagnostics.raw_count == 4
    assert result.diagnostics.retained_count == len(expected)


@pytest.mark.parametrize(
    ("top_k", "expected_count"),
    [(1, 1), (3, 3), (5, 3)],
)
def test_detection_processor_applies_global_top_k_after_filtering(
    top_k, expected_count
):
    detections = (
        detection(class_id=1, confidence=0.9),
        detection(class_id=2, confidence=0.8),
        detection(class_id=3, confidence=0.7),
    )
    processor = DetectionProcessor(
        DetectorDouble(detections),
        DetectionFilterSettings(top_k=top_k),
    )

    result = processor.process(frame())

    assert result.detections == detections[:expected_count]
    assert result.diagnostics.retained_count == expected_count


def test_detection_processor_returns_an_explicit_empty_result_after_filtering():
    processor = DetectionProcessor(
        DetectorDouble((detection(class_id=1),)),
        DetectionFilterSettings(selected_classes=(2,)),
    )

    result = processor.process(frame())

    assert result.detections == ()
    assert result.diagnostics.raw_count == 1
    assert result.diagnostics.retained_count == 0


def test_detection_processor_measures_provider_inference_with_its_clock():
    processor = DetectionProcessor(
        DetectorDouble((detection(),)), clock=iter((2.0, 2.125)).__next__
    )

    result = processor.process(frame())

    assert result.diagnostics.capture_duration is None
    assert result.diagnostics.inference_duration == 0.125
    assert result.diagnostics.processing_duration is None


def test_detection_processor_stops_its_timer_before_result_validation():
    clock = iter((2.0, 2.125)).__next__
    processor = DetectionProcessor(DetectorDouble([]), clock=clock)

    with pytest.raises(ValueError, match="Detection result"):
        processor.process(frame())

    with pytest.raises(StopIteration):
        clock()


def test_detection_processor_propagates_detector_errors_unchanged():
    error = RuntimeError("detection failed")
    processor = DetectionProcessor(DetectorDouble(error=error))

    with pytest.raises(RuntimeError) as raised:
        processor.process(frame())

    assert raised.value is error
