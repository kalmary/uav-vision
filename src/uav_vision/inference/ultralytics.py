import math
from numbers import Real
from typing import Any, Callable, Optional, Tuple

from uav_vision.config.models import model_path_for
from uav_vision.config.settings import InferenceSettings
from uav_vision.domain import BoundingBox, Detection, Frame

from .base import (
    InferenceInitializationError,
    InferenceResultError,
    InferenceRunError,
)


def _load_model(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path, task="detect")


def _tensor_values(tensor: Any) -> list:
    try:
        values = tensor.cpu().tolist()
    except Exception as error:
        raise InferenceResultError(
            "Model returned an invalid detection tensor"
        ) from error
    if not isinstance(values, list):
        raise InferenceResultError("Model returned an invalid detection tensor")
    return values


def _class_id(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
    ):
        raise InferenceResultError("Model returned an invalid class identifier")
    return int(value)


def _confidence(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise InferenceResultError("Model returned an invalid confidence")
    return float(value)


def _validate_names(names: Any) -> dict:
    if not isinstance(names, dict):
        raise InferenceResultError("Model returned invalid class names")
    for class_id, class_name in names.items():
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id < 0
            or not isinstance(class_name, str)
            or not class_name.strip()
        ):
            raise InferenceResultError("Model returned invalid class names")
    return names


class UltralyticsDetector:
    def __init__(
        self,
        settings: InferenceSettings,
        model_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        model_path = (
            str(settings.model_path)
            if settings.model_path is not None
            else model_path_for(settings.model_size)
        )
        self._device = settings.device
        try:
            self._model = (
                model_factory(model_path, task="detect")
                if model_factory is not None
                else _load_model(model_path)
            )
        except Exception as error:
            raise InferenceInitializationError(
                "Unable to load detection model"
            ) from error

    def detect(self, frame: Frame) -> Tuple[Detection, ...]:
        arguments = {"source": frame.image, "verbose": False}
        if self._device is not None:
            arguments["device"] = self._device
        try:
            results = self._model.predict(**arguments)
        except Exception as error:
            raise InferenceRunError("Unable to run object detection") from error
        if not isinstance(results, list) or len(results) != 1:
            raise InferenceResultError("Model must return one detection result")
        return self._detections_from_result(results[0])

    def _detections_from_result(self, result: Any) -> Tuple[Detection, ...]:
        try:
            boxes = result.boxes
            names = result.names
        except Exception as error:
            raise InferenceResultError(
                "Model returned an invalid detection result"
            ) from error
        if boxes is None:
            raise InferenceResultError("Model returned missing detection boxes")
        names = _validate_names(names)
        try:
            coordinates = _tensor_values(boxes.xyxy)
            class_ids = _tensor_values(boxes.cls)
            confidences = _tensor_values(boxes.conf)
        except AttributeError as error:
            raise InferenceResultError(
                "Model returned invalid detection boxes"
            ) from error
        if not (len(coordinates) == len(class_ids) == len(confidences)):
            raise InferenceResultError("Model returned mismatched detection fields")

        detections = []
        for index in range(len(coordinates)):
            detections.append(
                self._detection(
                    coordinates[index],
                    class_ids[index],
                    confidences[index],
                    names,
                )
            )
        return tuple(detections)

    @staticmethod
    def _detection(
        coordinates: Any,
        class_id_value: Any,
        confidence: Any,
        names: dict,
    ) -> Detection:
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise InferenceResultError("Model returned an invalid bounding box")
        class_id = _class_id(class_id_value)
        try:
            class_name = names[class_id]
        except (KeyError, TypeError) as error:
            raise InferenceResultError(
                "Model returned an unknown class identifier"
            ) from error
        if not isinstance(class_name, str) or not class_name.strip():
            raise InferenceResultError("Model returned an invalid class name")
        try:
            bounding_box = BoundingBox(*coordinates)
            return Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=_confidence(confidence),
                bounding_box=BoundingBox(
                    left=float(bounding_box.left),
                    top=float(bounding_box.top),
                    right=float(bounding_box.right),
                    bottom=float(bounding_box.bottom),
                ),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise InferenceResultError("Model returned an invalid detection") from error
