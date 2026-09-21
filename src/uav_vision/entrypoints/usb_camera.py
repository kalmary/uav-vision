from typing import Optional, Sequence

from uav_vision.config.settings import CameraType, CaptureSettings

from . import _run


def main(arguments: Optional[Sequence[str]] = None) -> int:
    return _run(
        arguments,
        prog="uav-vision-usb",
        capture_defaults=CaptureSettings(CameraType.OPENCV, 0),
    )
