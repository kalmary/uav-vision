from contextlib import ExitStack
from typing import Callable, Optional

from uav_vision.capture.gstreamer import GStreamerCamera
from uav_vision.capture.opencv import OpenCvCamera
from uav_vision.config.settings import AppSettings, CameraType
from uav_vision.inference.ultralytics import UltralyticsDetector
from uav_vision.output.log import LogOutput
from uav_vision.pipeline import run_pipeline
from uav_vision.processing.detection import DetectionProcessor


def run(
    settings: AppSettings,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    with ExitStack() as resources:
        log_output = LogOutput(settings.logging)
        resources.callback(log_output.close)
        log_output.startup(settings)

        try:
            if settings.capture.camera_type is CameraType.OPENCV:
                source = OpenCvCamera(settings.capture.source)
            else:
                source = GStreamerCamera(settings.capture.source)
        except BaseException as error:
            log_output.diagnostic("capture", "initialization failed", error)
            raise
        resources.callback(source.close)
        log_output.diagnostic(
            "capture",
            "initialized camera={0} source={1}".format(
                settings.capture.camera_type.value,
                settings.capture.source,
            ),
        )

        try:
            detector = UltralyticsDetector(settings.inference)
        except BaseException as error:
            log_output.diagnostic("inference", "initialization failed", error)
            raise
        log_output.diagnostic(
            "inference",
            "initialized provider=ultralytics model={0} device={1}".format(
                settings.inference.model_size.value
                if settings.inference.model_size is not None
                else settings.inference.model_path,
                settings.inference.device or "default",
            ),
        )

        try:
            processor = DetectionProcessor(detector)
        except BaseException as error:
            log_output.diagnostic("processing", "initialization failed", error)
            raise
        log_output.diagnostic("processing", "initialized type=detection")
        outputs = [log_output]

        if settings.display is not None:
            try:
                from uav_vision.output.display import DisplayOutput

                display_output = DisplayOutput(settings.display)
            except BaseException as error:
                log_output.diagnostic("display", "initialization failed", error)
                raise
            resources.callback(display_output.close)
            outputs.append(display_output)
            log_output.diagnostic(
                "display",
                "initialized dimensions={0}x{1}".format(
                    settings.display.width,
                    settings.display.height,
                ),
            )

        try:
            run_pipeline(source, processor, outputs, should_stop)
        except BaseException as error:
            log_output.diagnostic("pipeline", "failed", error)
            raise
