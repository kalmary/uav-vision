from typing import Any, Optional

from uav_vision.config.settings import DisplaySettings
from uav_vision.domain import ProcessedFrame


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

        resized = self._backend.resize(
            annotated,
            (self._settings.width, self._settings.height),
        )
        self._opened = True
        self._backend.imshow(self._window_name, resized)
        key = self._backend.waitKey(1)
        return (key & 0xFF) in (ord("q"), ord("Q"), 27)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._opened:
            self._opened = False
            self._backend.destroyWindow(self._window_name)
