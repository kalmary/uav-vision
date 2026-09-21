from contextlib import ExitStack
from typing import Callable, Optional

from uav_vision.capture.gstreamer import GStreamerCamera
from uav_vision.capture.opencv import OpenCvCamera
from uav_vision.config.settings import AppSettings, CameraType
from uav_vision.inference.ultralytics import UltralyticsDetector
from uav_vision.pipeline import run_pipeline
from uav_vision.processing.detection import DetectionProcessor


def run(
    settings: AppSettings,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    with ExitStack() as resources:
        if settings.capture.camera_type is CameraType.OPENCV:
            source = OpenCvCamera(settings.capture.source)
        else:
            source = GStreamerCamera(settings.capture.source)
        resources.callback(source.close)

        detector = UltralyticsDetector(settings.inference)
        processor = DetectionProcessor(detector)
        outputs = []

        if settings.display is not None:
            from uav_vision.output.display import DisplayOutput

            display_output = DisplayOutput(settings.display)
            resources.callback(display_output.close)
            outputs.append(display_output)

        run_pipeline(source, processor, outputs, should_stop)
