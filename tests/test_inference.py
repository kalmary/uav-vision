import math
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from uav_vision.config.models import ModelSize
from uav_vision.config.settings import InferenceSettings
from uav_vision.domain import DepthResult, Frame
from uav_vision.inference import (
    DepthEstimator,
    Detector,
    InferenceInitializationError,
    InferenceResultError,
    InferenceRunError,
    Segmenter,
    UltralyticsDepthEstimator,
    UltralyticsDetector,
    UltralyticsSegmenter,
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


class SemanticTensor:
    def __init__(self, value, cpu_error=None, numpy_error=None):
        self.value = value
        self.cpu_error = cpu_error
        self.numpy_error = numpy_error
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        if self.cpu_error is not None:
            raise self.cpu_error
        return self

    def numpy(self):
        if self.numpy_error is not None:
            raise self.numpy_error
        return self.value


class SemanticMask:
    def __init__(self, data):
        self.data = data


class SemanticResult:
    def __init__(self, class_map, names):
        self.semantic_mask = SemanticMask(SemanticTensor(class_map))
        self.names = names


class DepthTensor:
    def __init__(self, value, cpu_error=None, numpy_error=None):
        self.value = value
        self.cpu_error = cpu_error
        self.numpy_error = numpy_error
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        if self.cpu_error is not None:
            raise self.cpu_error
        return self

    def numpy(self):
        if self.numpy_error is not None:
            raise self.numpy_error
        return self.value


class DepthMap:
    def __init__(self, data):
        self.data = data


class DepthResultValue:
    def __init__(self, depth_map):
        self.depth = DepthMap(depth_map)


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
        captured_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )


def settings(device="cpu"):
    return InferenceSettings(model_size=ModelSize.NANO, device=device)


class SegmenterDouble:
    def segment(self, input_frame):
        return input_frame


class DepthEstimatorDouble:
    def estimate_depth(self, input_frame):
        return input_frame


def test_segmentation_and_depth_protocols_are_structurally_implemented():
    assert isinstance(SegmenterDouble(), Segmenter)
    assert isinstance(DepthEstimatorDouble(), DepthEstimator)


def make_detector(model, detector_settings=None, input_size=640):
    return UltralyticsDetector(
        detector_settings or settings(),
        "models/yolo26n.pt",
        input_size,
        model_factory=lambda path, task: model,
    )


def make_segmenter(model, segmenter_settings=None, input_size=640):
    return UltralyticsSegmenter(
        segmenter_settings or settings(),
        "models/yolo26n-sem.pt",
        input_size,
        model_factory=lambda path, task: model,
    )


def make_depth_estimator(model, estimator_settings=None, input_size=768):
    return UltralyticsDepthEstimator(
        estimator_settings or settings(),
        "models/yolo26n-depth.pt",
        input_size,
        model_factory=lambda path, task: model,
    )


def test_ultralytics_detector_structurally_implements_detector():
    detector = make_detector(Model())

    assert isinstance(detector, Detector)


def test_detector_loads_the_resolved_model_identifier():
    calls = []

    UltralyticsDetector(
        InferenceSettings(),
        "models/cached-detector.engine",
        640,
        model_factory=lambda path, task: calls.append((path, task)) or Model(),
    )

    assert calls == [("models/cached-detector.engine", "detect")]


def test_detector_requires_an_explicit_input_size():
    with pytest.raises(TypeError):
        UltralyticsDetector(
            InferenceSettings(),
            "models/yolo26n.pt",
            model_factory=lambda path, task: Model(),
        )


def test_model_is_loaded_once_during_initialization():
    calls = []
    model = Model([Result(Boxes([], [], []), {})])
    detector = UltralyticsDetector(
        settings(),
        "models/yolo26n.pt",
        640,
        model_factory=lambda path, task: calls.append((path, task)) or model,
    )

    detector.detect(frame())
    detector.detect(frame())

    assert calls == [("models/yolo26n.pt", "detect")]


def test_detect_uses_the_constructed_input_size_for_consecutive_frames():
    model = Model([Result(Boxes([], [], []), {})])
    detector = make_detector(model, input_size=512)
    input_frame = frame()

    detector.detect(input_frame)
    detector.detect(input_frame)

    assert [call["imgsz"] for call in model.calls] == [512, 512]


def test_detect_passes_the_frame_bgr_image_and_explicit_device():
    model = Model([Result(Boxes([], [], []), {})])
    detector = make_detector(model, settings(device="cuda"), input_size=512)
    input_frame = frame()

    assert detector.detect(input_frame) == ()
    assert model.calls == [
        {
            "source": input_frame.image,
            "verbose": False,
            "device": "cuda",
            "imgsz": 512,
        }
    ]


def test_detect_passes_the_cpu_device():
    model = Model([Result(Boxes([], [], []), {})])
    detector = make_detector(model)
    input_frame = frame()

    detector.detect(input_frame)

    assert model.calls == [
        {
            "source": input_frame.image,
            "verbose": False,
            "device": "cpu",
            "imgsz": 640,
        }
    ]


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
            "models/yolo26n.pt",
            640,
            model_factory=lambda path, task: (_ for _ in ()).throw(error),
        )

    assert raised.value.__cause__ is error


def test_initialization_wraps_missing_ultralytics_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(InferenceInitializationError) as raised:
        UltralyticsDetector(settings(), "models/yolo26n.pt", 640)

    assert isinstance(raised.value.__cause__, ImportError)


def test_detect_wraps_provider_errors_with_their_cause():
    error = RuntimeError("predict failure")
    detector = make_detector(Model(error=error))

    with pytest.raises(InferenceRunError) as raised:
        detector.detect(frame())

    assert raised.value.__cause__ is error


def test_ultralytics_segmenter_structurally_implements_segmenter():
    segmenter = make_segmenter(Model())

    assert isinstance(segmenter, Segmenter)


def test_segmenter_loads_the_resolved_model_for_semantic_inference():
    calls = []

    UltralyticsSegmenter(
        InferenceSettings(),
        "models/cached-segmenter.engine",
        640,
        model_factory=lambda path, task: calls.append((path, task)) or Model(),
    )

    assert calls == [("models/cached-segmenter.engine", "semantic")]


def test_segment_uses_the_frame_image_input_size_and_device_for_consecutive_frames():
    class_map = np.zeros((12, 16), dtype=np.int64)
    model = Model([SemanticResult(class_map, {0: "background"})])
    segmenter = make_segmenter(model, settings(device="cuda"), input_size=512)
    input_frame = frame()

    segmenter.segment(input_frame)
    segmenter.segment(input_frame)

    assert model.calls == [
        {
            "source": input_frame.image,
            "verbose": False,
            "imgsz": 512,
            "device": "cuda",
        },
        {
            "source": input_frame.image,
            "verbose": False,
            "imgsz": 512,
            "device": "cuda",
        },
    ]


def test_segment_converts_one_present_class_to_application_owned_data():
    class_map = np.full((12, 16), 3, dtype=np.int64)
    provider_result = SemanticResult(class_map, {1: "tree", 3: "road"})
    segmenter = make_segmenter(Model([provider_result]))

    result = segmenter.segment(frame())
    class_map[0, 0] = 1
    provider_result.names[3] = "changed"

    assert result.class_map.shape == (12, 16)
    assert result.class_map[0, 0] == 3
    assert [(item.class_id, item.class_name) for item in result.classes] == [
        (3, "road")
    ]
    assert result is not provider_result
    assert provider_result.semantic_mask.data.cpu_calls == 1


def test_segment_returns_sorted_metadata_for_only_the_present_classes():
    class_map = np.array([[4, 1] * 8] * 12, dtype=np.int32)
    segmenter = make_segmenter(
        Model([SemanticResult(class_map, {1: "tree", 2: "sky", 4: "road"})])
    )

    result = segmenter.segment(frame())

    assert [(item.class_id, item.class_name) for item in result.classes] == [
        (1, "tree"),
        (4, "road"),
    ]


def test_segment_wraps_initialization_errors_with_their_cause():
    error = RuntimeError("load failure")

    with pytest.raises(InferenceInitializationError) as raised:
        UltralyticsSegmenter(
            settings(),
            "models/yolo26n-sem.pt",
            640,
            model_factory=lambda path, task: (_ for _ in ()).throw(error),
        )

    assert raised.value.__cause__ is error


def test_segment_wraps_provider_errors_with_their_cause():
    error = RuntimeError("predict failure")
    segmenter = make_segmenter(Model(error=error))

    with pytest.raises(InferenceRunError) as raised:
        segmenter.segment(frame())

    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "results",
    [
        [],
        [
            SemanticResult(np.zeros((12, 16), dtype=np.int64), {0: "background"}),
            SemanticResult(np.zeros((12, 16), dtype=np.int64), {0: "background"}),
        ],
        (SemanticResult(np.zeros((12, 16), dtype=np.int64), {0: "background"}),),
    ],
)
def test_segment_requires_exactly_one_list_result(results):
    segmenter = make_segmenter(Model(results))

    with pytest.raises(InferenceResultError):
        segmenter.segment(frame())


@pytest.mark.parametrize(
    "result",
    [
        type("ResultWithoutSemanticMask", (), {"names": {0: "background"}})(),
        type(
            "ResultWithMissingSemanticMask",
            (),
            {"semantic_mask": None, "names": {0: "background"}},
        )(),
        type(
            "ResultWithoutSemanticData",
            (),
            {
                "semantic_mask": type("SemanticMaskWithoutData", (), {})(),
                "names": {0: "background"},
            },
        )(),
        type(
            "ResultWithMissingSemanticData",
            (),
            {
                "semantic_mask": type("SemanticMask", (), {"data": None})(),
                "names": {0: "background"},
            },
        )(),
    ],
)
def test_segment_rejects_missing_semantic_map(result):
    segmenter = make_segmenter(Model([result]))

    with pytest.raises(InferenceResultError):
        segmenter.segment(frame())


@pytest.mark.parametrize(
    "class_map",
    [
        np.zeros((12, 16), dtype=np.float32),
        np.zeros((1, 12, 16), dtype=np.int64),
        np.zeros((0, 16), dtype=np.int64),
        np.zeros((12, 15), dtype=np.int64),
        np.full((12, 16), -1, dtype=np.int64),
    ],
)
def test_segment_rejects_malformed_class_maps(class_map):
    segmenter = make_segmenter(Model([SemanticResult(class_map, {0: "background"})]))

    with pytest.raises(InferenceResultError):
        segmenter.segment(frame())


@pytest.mark.parametrize("names", [None, [], {0: ""}, {0: 123}, {"0": "road"}])
def test_segment_rejects_missing_or_invalid_class_names(names):
    segmenter = make_segmenter(
        Model([SemanticResult(np.zeros((12, 16), dtype=np.int64), names)])
    )

    with pytest.raises(InferenceResultError):
        segmenter.segment(frame())


def test_segment_rejects_unknown_class_identifiers():
    segmenter = make_segmenter(
        Model([SemanticResult(np.full((12, 16), 2, dtype=np.int64), {1: "tree"})])
    )

    with pytest.raises(InferenceResultError):
        segmenter.segment(frame())


@pytest.mark.parametrize("failure_method", ["cpu", "numpy"])
def test_segment_wraps_tensor_conversion_errors(failure_method):
    error = RuntimeError("tensor failure")
    tensor = SemanticTensor(
        np.zeros((12, 16), dtype=np.int64),
        cpu_error=error if failure_method == "cpu" else None,
        numpy_error=error if failure_method == "numpy" else None,
    )
    result = SemanticResult(np.zeros((12, 16), dtype=np.int64), {0: "background"})
    result.semantic_mask.data = tensor
    segmenter = make_segmenter(Model([result]))

    with pytest.raises(InferenceResultError) as raised:
        segmenter.segment(frame())

    assert raised.value.__cause__ is error


def test_ultralytics_depth_estimator_structurally_implements_depth_estimator():
    estimator = make_depth_estimator(Model())

    assert isinstance(estimator, DepthEstimator)


def test_depth_estimator_loads_the_resolved_model_for_depth_inference():
    calls = []

    UltralyticsDepthEstimator(
        settings(),
        "models/cached-depth.engine",
        768,
        model_factory=lambda path, task: calls.append((path, task)) or Model(),
    )

    assert calls == [("models/cached-depth.engine", "depth")]


def test_depth_estimator_requires_an_explicit_input_size():
    with pytest.raises(TypeError):
        UltralyticsDepthEstimator(
            settings(),
            "models/yolo26n-depth.pt",
            model_factory=lambda path, task: Model(),
        )


def test_estimate_depth_uses_frame_image_size_and_device_for_multiple_frames():
    depth_map = np.ones((12, 16), dtype=np.float32)
    model = Model([DepthResultValue(DepthTensor(depth_map))])
    estimator = make_depth_estimator(model, settings(device="cuda"), input_size=512)
    input_frame = frame()

    estimator.estimate_depth(input_frame)
    estimator.estimate_depth(input_frame)

    assert model.calls == [
        {
            "source": input_frame.image,
            "device": "cuda",
            "verbose": False,
            "imgsz": 512,
        },
        {
            "source": input_frame.image,
            "device": "cuda",
            "verbose": False,
            "imgsz": 512,
        },
    ]


@pytest.mark.parametrize("provider_uses_numpy", [False, True])
def test_estimate_depth_returns_application_owned_metric_depth(provider_uses_numpy):
    depth_map = np.arange(192, dtype=np.float32).reshape(12, 16)
    provider_map = depth_map if provider_uses_numpy else DepthTensor(depth_map)
    provider_result = DepthResultValue(provider_map)
    estimator = make_depth_estimator(Model([provider_result]))

    result = estimator.estimate_depth(frame())
    depth_map[0, 0] = 99.0

    assert isinstance(result, DepthResult)
    assert result is not provider_result
    assert result.depth_map[0, 0] == 0.0
    assert result.unit == "metre"
    assert result.scale == 1.0
    if not provider_uses_numpy:
        assert provider_map.cpu_calls == 1


def test_depth_initialization_wraps_model_factory_errors_with_their_cause():
    error = RuntimeError("load failure")

    with pytest.raises(InferenceInitializationError) as raised:
        UltralyticsDepthEstimator(
            settings(),
            "models/yolo26n-depth.pt",
            768,
            model_factory=lambda path, task: (_ for _ in ()).throw(error),
        )

    assert raised.value.__cause__ is error


def test_estimate_depth_wraps_provider_errors_with_their_cause():
    error = RuntimeError("predict failure")
    estimator = make_depth_estimator(Model(error=error))

    with pytest.raises(InferenceRunError) as raised:
        estimator.estimate_depth(frame())

    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "results",
    [
        [],
        [
            DepthResultValue(np.ones((12, 16), dtype=np.float32)),
            DepthResultValue(np.ones((12, 16), dtype=np.float32)),
        ],
        (DepthResultValue(np.ones((12, 16), dtype=np.float32)),),
    ],
)
def test_estimate_depth_requires_exactly_one_list_result(results):
    estimator = make_depth_estimator(Model(results))

    with pytest.raises(InferenceResultError):
        estimator.estimate_depth(frame())


@pytest.mark.parametrize(
    "result",
    [
        type("ResultWithoutDepth", (), {})(),
        type("ResultWithMissingDepth", (), {"depth": None})(),
        type("DepthWithoutData", (), {"depth": type("DepthMap", (), {})()})(),
        type("DepthWithMissingData", (), {"depth": DepthMap(None)})(),
    ],
)
def test_estimate_depth_rejects_missing_depth_data(result):
    estimator = make_depth_estimator(Model([result]))

    with pytest.raises(InferenceResultError):
        estimator.estimate_depth(frame())


def test_estimate_depth_wraps_results_property_errors_with_their_cause():
    error = RuntimeError("depth result failure")

    class FailingResult:
        @property
        def depth(self):
            raise error

    estimator = make_depth_estimator(Model([FailingResult()]))

    with pytest.raises(InferenceResultError) as raised:
        estimator.estimate_depth(frame())

    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "depth_map",
    [
        np.ones((1, 12, 16), dtype=np.float32),
        np.ones((0, 16), dtype=np.float32),
        np.ones((12, 15), dtype=np.float32),
        np.ones((12, 16), dtype=np.int32),
        np.full((12, 16), np.nan, dtype=np.float32),
        np.full((12, 16), np.inf, dtype=np.float32),
    ],
)
def test_estimate_depth_rejects_malformed_depth_maps(depth_map):
    estimator = make_depth_estimator(Model([DepthResultValue(depth_map)]))

    with pytest.raises(InferenceResultError):
        estimator.estimate_depth(frame())


@pytest.mark.parametrize("failure_method", ["cpu", "numpy"])
def test_estimate_depth_wraps_tensor_conversion_errors(failure_method):
    error = RuntimeError("depth tensor failure")
    tensor = DepthTensor(
        np.ones((12, 16), dtype=np.float32),
        cpu_error=error if failure_method == "cpu" else None,
        numpy_error=error if failure_method == "numpy" else None,
    )
    estimator = make_depth_estimator(Model([DepthResultValue(tensor)]))

    with pytest.raises(InferenceResultError) as raised:
        estimator.estimate_depth(frame())

    assert raised.value.__cause__ is error
