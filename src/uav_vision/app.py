from contextlib import ExitStack
from typing import Callable, Optional

from uav_vision.capture.gstreamer import GStreamerCamera
from uav_vision.capture.opencv import OpenCvCamera
from uav_vision.config.settings import AppSettings, CameraType, ProcessingType
from uav_vision.inference.ultralytics import (
    UltralyticsDetector,
    UltralyticsSegmenter,
)
from uav_vision.inference.ultralytics_depth import UltralyticsDepthEstimator
from uav_vision.output.log import LogOutput, attach_secondary_failure
from uav_vision.pipeline import run_pipeline
from uav_vision.processing.depth import DepthProcessor
from uav_vision.processing.detection import DetectionProcessor
from uav_vision.processing.segmentation import SegmentationProcessor


def _diagnose_failure(
    log_output: LogOutput,
    component: str,
    error: BaseException,
) -> None:
    try:
        log_output.diagnostic(component, "initialization failed", error)
    except BaseException as logging_error:
        attach_secondary_failure(error, logging_error)


def run(
    settings: AppSettings,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    processing_type = settings.processing.processing_type
    model = settings.model
    with ExitStack() as resources:
        log_output = LogOutput(settings.logging)
        resources.enter_context(log_output)
        log_output.startup(settings)

        try:
            if settings.capture.camera_type is CameraType.OPENCV:
                source = OpenCvCamera(settings.capture.source)
            else:
                source = GStreamerCamera(settings.capture.source)
        except BaseException as error:
            _diagnose_failure(log_output, "capture", error)
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
            if processing_type is ProcessingType.DETECTION:
                provider = UltralyticsDetector(
                    settings.inference,
                    model.identifier,
                    model.input_size,
                )
            elif processing_type is ProcessingType.SEGMENTATION:
                provider = UltralyticsSegmenter(
                    settings.inference,
                    model.identifier,
                    model.input_size,
                )
            else:
                provider = UltralyticsDepthEstimator(
                    settings.inference,
                    model.identifier,
                    model.input_size,
                )
        except BaseException as error:
            _diagnose_failure(log_output, "inference", error)
            raise
        log_output.diagnostic(
            "inference",
            "initialized provider=ultralytics model={0} device={1}".format(
                model.identifier,
                settings.inference.device,
            ),
        )

        try:
            if processing_type is ProcessingType.DETECTION:
                filters = (
                    settings.yolo.detection_filters
                    if settings.yolo is not None
                    else None
                )
                processor = (
                    DetectionProcessor(provider)
                    if filters is None
                    else DetectionProcessor(provider, filters)
                )
            elif processing_type is ProcessingType.SEGMENTATION:
                processor = SegmentationProcessor(provider)
            else:
                processor = DepthProcessor(provider)
        except BaseException as error:
            _diagnose_failure(log_output, "processing", error)
            raise
        log_output.diagnostic(
            "processing", "initialized type={}".format(processing_type.value)
        )
        outputs = [log_output]

        if settings.display.enabled:
            try:
                from uav_vision.output.display import DisplayOutput

                display_output = DisplayOutput(settings.display)
            except BaseException as error:
                _diagnose_failure(log_output, "display", error)
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
            run_pipeline(source, processor, outputs, should_stop, settings.fps)
        except BaseException as error:
            try:
                log_output.diagnostic("pipeline", "failed", error)
            except BaseException as logging_error:
                attach_secondary_failure(error, logging_error)
            raise
