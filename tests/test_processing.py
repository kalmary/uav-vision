from datetime import datetime, timezone

import numpy as np
import pytest

from uav_vision.config.settings import DetectionFilterSettings, ProcessingType
from uav_vision.domain import (
    BoundingBox,
    DepthResult,
    Detection,
    Frame,
    SegmentationClass,
    SegmentationResult,
)
from uav_vision.processing import (
    DepthProcessor,
    DetectionProcessor,
    FrameProcessor,
    SegmentationProcessor,
)


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


class SegmenterDouble:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def segment(self, input_frame):
        self.calls.append(input_frame)
        if self.error is not None:
            raise self.error
        return self.result


class DepthEstimatorDouble:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def estimate_depth(self, input_frame):
        self.calls.append(input_frame)
        if self.error is not None:
            raise self.error
        return self.result


def frame():
    return Frame(
        image=np.zeros((12, 16, 3), dtype=np.uint8),
        captured_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )


def detection(class_id=3, confidence=0.75):
    return Detection(
        class_id=class_id,
        class_name="class-{}".format(class_id),
        confidence=confidence,
        bounding_box=BoundingBox(1.0, 2.0, 8.0, 10.0),
    )


def segmentation_result(shape=(12, 16), classes=None):
    classes = classes or (SegmentationClass(0, "background"),)
    return SegmentationResult(np.zeros(shape, dtype=np.int64), classes)


def depth_result(shape=(12, 16)):
    return DepthResult(np.ones(shape, dtype=np.float32), "metre", 1.0)


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


def test_segmentation_processor_structurally_implements_frame_processor():
    processor = SegmentationProcessor(SegmenterDouble(segmentation_result()))

    assert isinstance(processor, FrameProcessor)


def test_segmentation_processor_returns_the_same_frame_and_segmentation_result():
    expected = segmentation_result()
    segmenter = SegmenterDouble(expected)
    processor = SegmentationProcessor(segmenter)
    input_frame = frame()

    processed = processor.process(input_frame)

    assert segmenter.calls == [input_frame]
    assert processed.frame is input_frame
    assert processed.result is expected
    assert processed.processing_type is ProcessingType.SEGMENTATION


def test_segmentation_processor_counts_present_classes_in_diagnostics():
    expected = segmentation_result(
        classes=(
            SegmentationClass(0, "background"),
            SegmentationClass(2, "tree"),
        )
    )
    processor = SegmentationProcessor(SegmenterDouble(expected))

    processed = processor.process(frame())

    assert processed.diagnostics.raw_count == 2
    assert processed.diagnostics.retained_count == 2


def test_segmentation_processor_measures_only_provider_inference():
    clock_values = iter((3.0, 3.25))
    processor = SegmentationProcessor(
        SegmenterDouble(segmentation_result()), clock=clock_values.__next__
    )

    processed = processor.process(frame())

    assert processed.diagnostics.capture_duration is None
    assert processed.diagnostics.inference_duration == 0.25
    assert processed.diagnostics.processing_duration is None
    with pytest.raises(StopIteration):
        clock_values.__next__()


def test_segmentation_processor_stops_its_timer_before_result_validation():
    clock_values = iter((4.0, 4.125))
    processor = SegmentationProcessor(
        SegmenterDouble(segmentation_result(shape=(1, 1))),
        clock=clock_values.__next__,
    )

    with pytest.raises(ValueError, match="dimensions"):
        processor.process(frame())

    with pytest.raises(StopIteration):
        clock_values.__next__()


def test_segmentation_processor_propagates_segmenter_errors_unchanged():
    error = RuntimeError("segmentation failed")
    processor = SegmentationProcessor(SegmenterDouble(error=error))

    with pytest.raises(RuntimeError) as raised:
        processor.process(frame())

    assert raised.value is error


def test_depth_processor_structurally_implements_frame_processor():
    processor = DepthProcessor(DepthEstimatorDouble(depth_result()))

    assert isinstance(processor, FrameProcessor)


def test_depth_processor_returns_the_same_frame_and_depth_result():
    expected = depth_result()
    estimator = DepthEstimatorDouble(expected)
    processor = DepthProcessor(estimator)
    input_frame = frame()

    processed = processor.process(input_frame)

    assert estimator.calls == [input_frame]
    assert processed.frame is input_frame
    assert processed.result is expected
    assert processed.processing_type is ProcessingType.DEPTH
    assert processed.diagnostics.raw_count == expected.depth_map.size
    assert processed.diagnostics.retained_count == expected.depth_map.size


def test_depth_processor_measures_only_provider_inference():
    clock_values = iter((5.0, 5.375))
    processor = DepthProcessor(
        DepthEstimatorDouble(depth_result()), clock=clock_values.__next__
    )

    processed = processor.process(frame())

    assert processed.diagnostics.capture_duration is None
    assert processed.diagnostics.inference_duration == 0.375
    assert processed.diagnostics.processing_duration is None
    with pytest.raises(StopIteration):
        clock_values.__next__()


def test_depth_processor_propagates_estimator_errors_unchanged():
    error = RuntimeError("depth estimation failed")
    clock_values = iter((7.0, 8.0))
    processor = DepthProcessor(
        DepthEstimatorDouble(error=error), clock=clock_values.__next__
    )

    with pytest.raises(RuntimeError) as raised:
        processor.process(frame())

    assert raised.value is error
    assert clock_values.__next__() == 8.0
