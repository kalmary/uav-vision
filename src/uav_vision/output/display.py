from typing import Any, Optional, Tuple

import numpy as np

from uav_vision.config.settings import DisplaySettings
from uav_vision.domain import (
    DepthResult,
    DetectionResult,
    ProcessedFrame,
    SegmentationResult,
)

from .colors import segmentation_color


def _fit_text(
    backend: Any,
    text: str,
    font: int,
    max_width: int,
    max_height: int,
    thickness: int = 3,
) -> Tuple[float, int, int]:
    scale = 0.5
    for _ in range(8):
        (width, height), baseline = backend.getTextSize(text, font, scale, thickness)
        if width <= max_width and height + baseline <= max_height:
            return scale, height, baseline
        ratio = min(max_width / width, max_height / (height + baseline))
        scale *= min(ratio, 0.9)
    (_, height), baseline = backend.getTextSize(text, font, scale, thickness)
    return scale, height, baseline


def _draw_fps(backend: Any, image: np.ndarray, text: str) -> None:
    image_height, image_width = image.shape[:2]
    padding = max(
        0,
        min(4, (image_width - 1) // 2, (image_height - 1) // 2),
    )
    scale, text_height, _ = _fit_text(
        backend,
        text,
        backend.FONT_HERSHEY_SIMPLEX,
        max(1, image_width - 2 * padding),
        max(1, image_height - 2 * padding),
        thickness=1,
    )
    backend.putText(
        image,
        text,
        (padding, padding + text_height),
        backend.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
    )


class DisplayOutput:
    def __init__(
        self,
        settings: DisplaySettings,
        backend: Optional[Any] = None,
        window_name: str = "UAV Vision",
    ) -> None:
        if backend is None:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("OpenCV is required for display output") from error
            backend = cv2

        self._settings = settings
        self._backend = backend
        self._window_name = window_name
        self._closed = False
        self._opened = False

    def write(self, processed: ProcessedFrame) -> bool:
        if self._closed:
            raise RuntimeError("Display output is closed")

        annotated = processed.frame.image.copy()
        if isinstance(processed.result, SegmentationResult):
            overlay = np.empty_like(annotated)
            for class_id in np.unique(processed.result.class_map):
                overlay[processed.result.class_map == class_id] = segmentation_color(
                    int(class_id)
                )
            annotated = (
                (annotated.astype(np.uint16) * 3 + overlay.astype(np.uint16) * 2) // 5
            ).astype(np.uint8)
        elif isinstance(processed.result, DepthResult):
            minimum = processed.result.minimum
            maximum = processed.result.maximum
            if minimum == maximum:
                normalized = np.zeros(processed.result.depth_map.shape, dtype=np.uint8)
            else:
                values = processed.result.depth_map.astype(np.float64, copy=True)
                span = maximum - minimum
                if np.isfinite(span):
                    values = (values - minimum) / span * 255.0
                else:
                    magnitude = max(abs(minimum), abs(maximum))
                    lower = minimum / magnitude
                    values = (values / magnitude - lower) * (
                        255.0 / (maximum / magnitude - lower)
                    )
                normalized = np.clip(values, 0.0, 255.0).astype(np.uint8)
            annotated = self._backend.applyColorMap(
                normalized,
                self._backend.COLORMAP_VIRIDIS,
            )

        frames_per_second = processed.diagnostics.frames_per_second
        fps = "--" if frames_per_second is None else "{0:.2f}".format(frames_per_second)
        if isinstance(processed.result, DetectionResult):
            self._backend.putText(
                annotated,
                "FPS: " + fps,
                (10, 20),
                self._backend.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            for detection in processed.detections:
                box = detection.bounding_box
                start = (int(round(box.left)), int(round(box.top)))
                end = (int(round(box.right)), int(round(box.bottom)))
                self._backend.rectangle(annotated, start, end, (0, 255, 0), 2)
                self._backend.putText(
                    annotated,
                    "{0} {1:.2f}".format(
                        detection.class_name,
                        detection.confidence,
                    ),
                    (start[0], max(start[1] - 10, 0)),
                    self._backend.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        scale = min(
            self._settings.width / processed.frame.width,
            self._settings.height / processed.frame.height,
        )
        width = min(
            self._settings.width,
            max(1, int(round(processed.frame.width * scale))),
        )
        height = min(
            self._settings.height,
            max(1, int(round(processed.frame.height * scale))),
        )
        resized = self._backend.resize(annotated, (width, height))
        displayed = np.zeros(
            (self._settings.height, self._settings.width, 3),
            dtype=annotated.dtype,
        )
        left = (self._settings.width - width) // 2
        top = (self._settings.height - height) // 2
        displayed[top : top + height, left : left + width] = resized
        if isinstance(processed.result, (SegmentationResult, DepthResult)):
            _draw_fps(self._backend, displayed, "FPS: " + fps)
        self._opened = True
        self._backend.imshow(self._window_name, displayed)
        key = self._backend.waitKey(1)
        return (key & 0xFF) in (ord("q"), ord("Q"), 27)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._opened:
            self._opened = False
            self._backend.destroyWindow(self._window_name)
