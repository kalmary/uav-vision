from datetime import datetime, timezone

import numpy as np
import pytest

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


def detection():
    return Detection(
        class_id=3,
        class_name="car",
        confidence=0.75,
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
