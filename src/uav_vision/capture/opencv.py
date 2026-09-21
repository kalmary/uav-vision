from typing import Any, Callable, Optional, Type

from uav_vision.domain import Frame

from .base import CaptureInitializationError, CaptureReadError, frame_from_image


def _open_video_capture(source: Any) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise CaptureInitializationError(
            "OpenCV is required for camera capture"
        ) from error
    return cv2.VideoCapture(source)


class OpenCvCamera:
    def __init__(
        self,
        source: Any,
        capture_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if (
            isinstance(source, bool)
            or (isinstance(source, int) and source < 0)
            or (isinstance(source, str) and not source.strip())
            or not isinstance(source, (int, str))
        ):
            raise ValueError(
                "camera source must be a non-negative index or non-empty string"
            )

        self._capture: Optional[Any] = None
        self._closed = False
        self._sequence = 0
        self._is_file_source = isinstance(source, str) and "://" not in source

        try:
            self._capture = (capture_factory or _open_video_capture)(source)
            opened = self._capture.isOpened()
        except Exception as error:
            self.close()
            raise CaptureInitializationError("Unable to open camera source") from error
        if not opened:
            self.close()
            raise CaptureInitializationError("Unable to open camera source")

    def read(self) -> Optional[Frame]:
        if self._closed or self._capture is None:
            raise CaptureReadError("Camera is closed")
        try:
            succeeded, image = self._capture.read()
        except Exception as error:
            raise CaptureReadError("Unable to read camera frame") from error
        if not succeeded:
            if self._is_file_source:
                return None
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

    def __enter__(self) -> "OpenCvCamera":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        self.close()
