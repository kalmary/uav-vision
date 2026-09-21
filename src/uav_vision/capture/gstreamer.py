from typing import Any, Callable, Optional, Type

from uav_vision.domain import Frame

from .base import CaptureInitializationError, CaptureReadError, frame_from_image


def _open_video_capture(pipeline: str) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise CaptureInitializationError(
            "OpenCV is required for camera capture"
        ) from error
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


class GStreamerCamera:
    def __init__(
        self,
        pipeline: str,
        capture_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if not isinstance(pipeline, str) or not pipeline.strip():
            raise ValueError("GStreamer pipeline must be a non-empty string")

        self._capture: Optional[Any] = None
        self._closed = False
        self._sequence = 0

        try:
            if capture_factory is None:
                self._capture = _open_video_capture(pipeline)
            else:
                self._capture = capture_factory(pipeline)
            opened = self._capture.isOpened()
        except Exception as error:
            self.close()
            raise CaptureInitializationError(
                "Unable to open GStreamer pipeline"
            ) from error
        if not opened:
            self.close()
            raise CaptureInitializationError("Unable to open GStreamer pipeline")

    def read(self) -> Optional[Frame]:
        if self._closed or self._capture is None:
            raise CaptureReadError("Camera is closed")
        try:
            succeeded, image = self._capture.read()
        except Exception as error:
            raise CaptureReadError("Unable to read camera frame") from error
        if not succeeded:
            raise CaptureReadError("Unable to read camera frame")

        frame = frame_from_image(image, self._sequence)
        self._sequence += 1
        return frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                self._capture = None

    def __enter__(self) -> "GStreamerCamera":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        self.close()
