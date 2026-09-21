import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from uav_vision.config.models import ModelSize, model_path_for
from uav_vision.config.settings import InferenceSettings
from uav_vision.domain import Frame
from uav_vision.inference import (
    Detector,
    InferenceInitializationError,
    InferenceResultError,
    InferenceRunError,
    UltralyticsDetector,
)


class Tensor:
    def __init__(self, value, error=None):
        self.value = value
        self.error = error

    def cpu(self):
        if self.error is not None:
            raise self.error
        return self

    def tolist(self):
        return self.value


class Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = Tensor(xyxy)
        self.cls = Tensor(cls)
        self.conf = Tensor(conf)


class Result:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class Model:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


def frame():
    return Frame(
        image=np.zeros((12, 16, 3), dtype=np.uint8),
        sequence=4,
        captured_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )


def settings(device=None):
    return InferenceSettings(model_size=ModelSize.NANO, device=device)


def make_detector(model, detector_settings=None):
    return UltralyticsDetector(
        detector_settings or settings(),
        model_factory=lambda path, task: model,
    )


def test_ultralytics_detector_structurally_implements_detector():
    detector = make_detector(Model())

    assert isinstance(detector, Detector)


@pytest.mark.parametrize("model_size", list(ModelSize))
def test_model_size_uses_central_model_path_mapping(model_size):
    calls = []
    detector_settings = InferenceSettings(model_size=model_size)

    UltralyticsDetector(
        detector_settings,
        model_factory=lambda path, task: calls.append((path, task)) or Model(),
    )

    assert calls == [(model_path_for(model_size), "detect")]


@pytest.mark.parametrize("model_path", [Path("custom.pt"), Path("custom.engine")])
def test_explicit_model_path_bypasses_model_size_alias(model_path):
    calls = []
    detector_settings = InferenceSettings(model_size=None, model_path=model_path)

    UltralyticsDetector(
        detector_settings,
        model_factory=lambda path, task: calls.append((path, task)) or Model(),
    )

    assert calls == [(str(model_path), "detect")]


def test_model_is_loaded_once_during_initialization():
    calls = []
    model = Model([Result(Boxes([], [], []), {})])
    detector = UltralyticsDetector(
        settings(),
        model_factory=lambda path, task: calls.append((path, task)) or model,
    )

    detector.detect(frame())
    detector.detect(frame())

    assert calls == [(model_path_for(ModelSize.NANO), "detect")]


def test_detect_passes_the_frame_bgr_image_and_explicit_device():
    model = Model([Result(Boxes([], [], []), {})])
    detector = make_detector(model, settings(device="cuda:0"))
    input_frame = frame()

    assert detector.detect(input_frame) == ()
    assert model.calls == [
        {"source": input_frame.image, "verbose": False, "device": "cuda:0"}
    ]


def test_detect_omits_device_when_it_is_not_selected():
    model = Model([Result(Boxes([], [], []), {})])
    detector = make_detector(model)
    input_frame = frame()

    detector.detect(input_frame)

    assert model.calls == [{"source": input_frame.image, "verbose": False}]


def test_detect_returns_no_detections_for_empty_boxes():
    detector = make_detector(Model([Result(Boxes([], [], []), {})]))

    assert detector.detect(frame()) == ()


def test_detect_converts_one_detection():
    detector = make_detector(
        Model([Result(Boxes([[1, 2, 8, 10]], [3.0], [0.75]), {3: "car"})])
    )

    detections = detector.detect(frame())

    assert len(detections) == 1
    assert detections[0].class_id == 3
    assert detections[0].class_name == "car"
    assert detections[0].confidence == 0.75
    assert detections[0].bounding_box.left == 1.0
    assert detections[0].bounding_box.top == 2.0
    assert detections[0].bounding_box.right == 8.0
    assert detections[0].bounding_box.bottom == 10.0
    assert all(
        isinstance(value, float)
        for value in (
            detections[0].bounding_box.left,
            detections[0].bounding_box.top,
            detections[0].bounding_box.right,
            detections[0].bounding_box.bottom,
        )
    )


def test_detect_preserves_multiple_detection_order():
    detector = make_detector(
        Model(
            [
                Result(
                    Boxes([[1, 2, 8, 10], [3.5, 4.5, 12, 11]], [3.0, 1.0], [0.75, 0.2]),
                    {1: "bicycle", 3: "car"},
                )
            ]
        )
    )

    detections = detector.detect(frame())

    detection_values = [
        (item.class_id, item.class_name, item.confidence) for item in detections
    ]
    assert detection_values == [
        (3, "car", 0.75),
        (1, "bicycle", 0.2),
    ]
    assert detections[1].bounding_box.left == 3.5


@pytest.mark.parametrize("class_id", [1.5, -1.0, math.inf, math.nan])
def test_detect_rejects_invalid_class_identifier(class_id):
    detector = make_detector(
        Model([Result(Boxes([[1, 2, 8, 10]], [class_id], [0.75]), {1: "bicycle"})])
    )

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


def test_detect_rejects_an_unknown_class_identifier():
    detector = make_detector(
        Model([Result(Boxes([[1, 2, 8, 10]], [4.0], [0.75]), {3: "car"})])
    )

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


@pytest.mark.parametrize("names", [None, [], {3: ""}, {3: 123}, {"3": "car"}])
def test_detect_rejects_missing_or_invalid_class_names(names):
    detector = make_detector(
        Model([Result(Boxes([[1, 2, 8, 10]], [3.0], [0.75]), names)])
    )

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


@pytest.mark.parametrize(
    ("xyxy", "cls", "conf"),
    [
        ([[1, 2, 8, 10]], [], [0.75]),
        ([[1, 2, 8, 10]], [3.0], []),
        ([], [3.0], [0.75]),
    ],
)
def test_detect_rejects_box_tensor_length_mismatches(xyxy, cls, conf):
    detector = make_detector(Model([Result(Boxes(xyxy, cls, conf), {3: "car"})]))

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


@pytest.mark.parametrize(
    "bbox",
    [
        [1, 2, 8],
        [1, 2, 8, 10, 11],
        [1, 2, math.inf, 10],
        [1, 2, 1, 10],
        [-1, 2, 8, 10],
        [1, 2, "8", 10],
    ],
)
def test_detect_rejects_invalid_bounding_boxes(bbox):
    detector = make_detector(Model([Result(Boxes([bbox], [3.0], [0.75]), {3: "car"})]))

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.inf, math.nan, "0.75"])
def test_detect_rejects_invalid_confidence(confidence):
    detector = make_detector(
        Model([Result(Boxes([[1, 2, 8, 10]], [3.0], [confidence]), {3: "car"})])
    )

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


@pytest.mark.parametrize(
    "results",
    [
        [],
        [Result(Boxes([], [], []), {}), Result(Boxes([], [], []), {})],
        (Result(Boxes([], [], []), {}),),
    ],
)
def test_detect_requires_exactly_one_list_result(results):
    detector = make_detector(Model(results))

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


def test_detect_rejects_none_boxes():
    detector = make_detector(Model([Result(None, {})]))

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


def test_detect_rejects_results_without_boxes():
    result = type("ResultWithoutBoxes", (), {"names": {}})()
    detector = make_detector(Model([result]))

    with pytest.raises(InferenceResultError) as raised:
        detector.detect(frame())

    assert isinstance(raised.value.__cause__, AttributeError)


def test_detect_rejects_tensors_that_do_not_convert_to_lists():
    boxes = Boxes([], [], [])
    boxes.xyxy = Tensor(())
    detector = make_detector(Model([Result(boxes, {})]))

    with pytest.raises(InferenceResultError):
        detector.detect(frame())


def test_detect_wraps_tensor_conversion_errors_with_their_cause():
    boxes = Boxes([], [], [])
    error = RuntimeError("tensor failure")
    boxes.xyxy = Tensor([], error=error)
    detector = make_detector(Model([Result(boxes, {})]))

    with pytest.raises(InferenceResultError) as raised:
        detector.detect(frame())

    assert raised.value.__cause__ is error


def test_initialization_wraps_model_factory_errors_with_their_cause():
    error = RuntimeError("load failure")

    with pytest.raises(InferenceInitializationError) as raised:
        UltralyticsDetector(
            settings(),
            model_factory=lambda path, task: (_ for _ in ()).throw(error),
        )

    assert raised.value.__cause__ is error


def test_initialization_wraps_missing_ultralytics_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(InferenceInitializationError) as raised:
        UltralyticsDetector(settings())

    assert isinstance(raised.value.__cause__, ImportError)


def test_detect_wraps_provider_errors_with_their_cause():
    error = RuntimeError("predict failure")
    detector = make_detector(Model(error=error))

    with pytest.raises(InferenceRunError) as raised:
        detector.detect(frame())

    assert raised.value.__cause__ is error
