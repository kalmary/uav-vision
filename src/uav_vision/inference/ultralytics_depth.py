from typing import Any, Callable, Optional

import numpy as np

from uav_vision.config.settings import InferenceSettings
from uav_vision.domain import DepthResult, Frame

from .base import (
    InferenceInitializationError,
    InferenceResultError,
    InferenceRunError,
)


def _load_model(model_identifier: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_identifier, task="depth")


def _depth_map(data: Any, frame: Frame) -> np.ndarray:
    if isinstance(data, np.ndarray):
        values = data
    else:
        try:
            values = data.cpu().numpy()
        except Exception as error:
            raise InferenceResultError(
                "Model returned an invalid depth tensor"
            ) from error
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.size == 0
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
        or values.shape != frame.image.shape[:2]
    ):
        raise InferenceResultError("Model returned an invalid depth map")
    return values


class UltralyticsDepthEstimator:
    def __init__(
        self,
        settings: InferenceSettings,
        model_identifier: str,
        input_size: int,
        model_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._device = settings.device
        self._input_size = input_size
        try:
            self._model = (
                model_factory(model_identifier, task="depth")
                if model_factory is not None
                else _load_model(model_identifier)
            )
        except Exception as error:
            raise InferenceInitializationError(
                "Unable to load depth estimation model"
            ) from error

    def estimate_depth(self, frame: Frame) -> DepthResult:
        arguments = {
            "source": frame.image,
            "device": self._device,
            "verbose": False,
            "imgsz": self._input_size,
        }
        try:
            results = self._model.predict(**arguments)
        except Exception as error:
            raise InferenceRunError("Unable to run depth estimation") from error
        if not isinstance(results, list) or len(results) != 1:
            raise InferenceResultError("Model must return one depth result")
        return self._depth_from_result(results[0], frame)

    @staticmethod
    def _depth_from_result(result: Any, frame: Frame) -> DepthResult:
        try:
            depth = result.depth
        except Exception as error:
            raise InferenceResultError(
                "Model returned an invalid depth result"
            ) from error
        if depth is None:
            raise InferenceResultError("Model returned a missing depth map")
        try:
            data = depth.data
        except Exception as error:
            raise InferenceResultError("Model returned an invalid depth map") from error
        if data is None:
            raise InferenceResultError("Model returned a missing depth map")
        return DepthResult(_depth_map(data, frame), "metre", 1.0)
