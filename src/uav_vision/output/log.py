import re
import sys
import traceback
from typing import Optional, TextIO

import numpy as np

from uav_vision.config.settings import AppSettings, LogLevel, LogSettings
from uav_vision.domain import (
    DepthResult,
    DetectionResult,
    ProcessedFrame,
    SegmentationResult,
)

from .colors import nearest_color_name, segmentation_color

_SOURCE_CREDENTIAL = re.compile(
    r"(?i)(?P<prefix>(?:^|[?&;,\s])(?:user(?:[-_]?id|name)?|user[-_]?pw|"
    r"password|passwd|token|access[-_]?token|auth[-_]?token)\s*(?:=|:)\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s&;,#]+)"
)


def _is_uri_userinfo(value: str) -> bool:
    separator = value.find(":")
    if separator < 0:
        return not any(character in "/?# \t\r\n" for character in value)
    if any(character in "/?#" for character in value[:separator]):
        return False

    suffix = value[separator + 1 :]
    digits = 0
    while digits < len(suffix) and suffix[digits].isdigit():
        digits += 1
    if digits > 0 and digits < len(suffix) and suffix[digits] in "/?#":
        return not any(character in " \t\r\n" for character in suffix[digits + 1 :])
    return not (digits > 0 and digits < len(suffix) and suffix[digits] in " \t\r\n")


def _uri_userinfo_end(value: str, authority_start: int, segment_end: int) -> int:
    credential_end = value.find("@", authority_start, segment_end)
    if credential_end < 0 or not _is_uri_userinfo(
        value[authority_start:credential_end]
    ):
        return -1

    while True:
        next_end = value.find("@", credential_end + 1, segment_end)
        if next_end < 0:
            return credential_end
        between = value[credential_end + 1 : next_end]
        if any(character in "/?#" for character in between):
            return credential_end
        credential_end = next_end


def _redact_text(value: str) -> str:
    schemes = []
    index = 0
    while index < len(value):
        if not ("A" <= value[index] <= "Z" or "a" <= value[index] <= "z"):
            index += 1
            continue
        end = index + 1
        while end < len(value) and (
            "A" <= value[end] <= "Z"
            or "a" <= value[end] <= "z"
            or "0" <= value[end] <= "9"
            or value[end] in "+.-"
        ):
            end += 1
        if value[end : end + 3] == "://":
            schemes.append((index, end + 3))
            index = end + 3
            continue
        index += 1

    redacted = []
    start = 0
    for index, (_, authority_start) in enumerate(schemes):
        segment_end = schemes[index + 1][0] if index + 1 < len(schemes) else len(value)
        credential_end = _uri_userinfo_end(value, authority_start, segment_end)
        if credential_end < 0:
            continue
        redacted.append(value[start:authority_start])
        redacted.append("***@")
        start = credential_end + 1
    redacted.append(value[start:])
    return _SOURCE_CREDENTIAL.sub(r"\g<prefix>***", "".join(redacted))


def attach_secondary_failure(primary: BaseException, secondary: BaseException) -> None:
    previous = primary.__context__
    secondary.__context__ = previous
    primary.__context__ = secondary


class LogOutput:
    def __init__(self, settings: LogSettings, stream: Optional[TextIO] = None) -> None:
        self._settings = settings
        self._closed = False
        self._started = False
        self._owns_stream = stream is None and settings.path is not None
        if stream is not None:
            self._stream = stream
        elif settings.path is not None:
            self._stream = settings.path.open("a", encoding="utf-8")
        else:
            self._stream = sys.stderr

    def startup(self, settings: AppSettings) -> None:
        if self._closed:
            raise RuntimeError("Log output is closed")
        if self._started:
            raise RuntimeError("Startup configuration was already written")
        model = settings.model
        if settings.yolo is None:
            raise ValueError("YOLO configuration is required for startup logging")
        self._started = True
        filters = settings.yolo.detection_filters
        selected_classes = (
            "all"
            if not filters.selected_classes
            else ",".join(str(class_id) for class_id in filters.selected_classes)
        )
        top_k = "unlimited" if filters.top_k is None else str(filters.top_k)
        source_name = "index" if isinstance(settings.capture.source, int) else "source"
        destination = (
            "console" if self._settings.path is None else str(self._settings.path)
        )
        self._write(
            (
                "startup camera.type={0} camera.{1}={2} processing.type={3} "
                "inference.model_size={4} inference.model={5} "
                "inference.input_size={6} inference.device={7} runtime.fps={8} "
                "detection.selected_classes={9} "
                "detection.minimum_confidence={10} detection.top_k={11} "
                "display.enabled={12} display.width={13} display.height={14} "
                "logging.level={15} logging.destination={16} yolo.origin={17}"
            ).format(
                settings.capture.camera_type.value,
                source_name,
                settings.capture.source,
                settings.processing.processing_type.value,
                settings.inference.model_size.value,
                model.identifier,
                model.input_size,
                settings.inference.device,
                settings.fps,
                selected_classes,
                filters.minimum_confidence,
                top_k,
                str(settings.display.enabled).lower(),
                settings.display.width,
                settings.display.height,
                self._settings.level.value,
                destination,
                settings.yolo.origin,
            )
        )

    def diagnostic(
        self,
        component: str,
        event: str,
        error: Optional[BaseException] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Log output is closed")
        if self._settings.level is not LogLevel.DEBUG and error is None:
            return
        record = "diagnostic component={0} event={1}".format(component, event)
        if error is not None:
            record = record + " error={0}: {1}".format(
                type(error).__name__,
                error,
            )
            if self._settings.level is LogLevel.DEBUG:
                details = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ).rstrip()
                record = record + "\n" + details
        self._write(record)

    def write(self, processed: ProcessedFrame) -> bool:
        if self._closed:
            raise RuntimeError("Log output is closed")
        frames_per_second = processed.diagnostics.frames_per_second
        fps = (
            "unavailable"
            if frames_per_second is None
            else "{0:.2f}".format(frames_per_second)
        )
        if isinstance(processed.result, DetectionResult):
            detections = ", ".join(
                "class={0} confidence={1:.6f} bbox=({2},{3},{4},{5})".format(
                    detection.class_name,
                    detection.confidence,
                    detection.bounding_box.left,
                    detection.bounding_box.top,
                    detection.bounding_box.right,
                    detection.bounding_box.bottom,
                )
                for detection in processed.result.detections
            )
            self._write(
                "frame fps={0} detections=[{1}]".format(
                    fps,
                    detections,
                )
            )
        elif isinstance(processed.result, SegmentationResult):
            present_classes = np.unique(processed.result.class_map)
            class_names = {
                value.class_id: value.class_name for value in processed.result.classes
            }
            labels = []
            for value in present_classes:
                class_id = int(value)
                blue, green, red = segmentation_color(class_id)
                labels.append(
                    "{0}: {1} color={2}".format(
                        class_id,
                        class_names[class_id],
                        nearest_color_name((red, green, blue)),
                    )
                )
            self._write(
                (
                    "frame fps={0} segmentation.classes={1} segmentation.labels=[{2}]"
                ).format(
                    fps,
                    len(present_classes),
                    ", ".join(labels),
                )
            )
        elif isinstance(processed.result, DepthResult):
            self._write(
                (
                    "frame fps={0} depth.minimum={1:.6f} "
                    "depth.maximum={2:.6f} depth.mean={3:.6f} "
                    "depth.standard_deviation={4:.6f} depth.unit={5} "
                    "depth.scale={6}"
                ).format(
                    fps,
                    processed.result.minimum,
                    processed.result.maximum,
                    processed.result.mean,
                    processed.result.standard_deviation,
                    processed.result.unit,
                    processed.result.scale,
                )
            )
        if self._settings.level is LogLevel.DEBUG:
            diagnostics = processed.diagnostics
            record = (
                "frame captured_at={0} dimensions={1}x{2} "
                "raw_count={3} retained_count={4} capture_duration={5:.6f} "
                "inference_duration={6:.6f} processing_duration={7:.6f}".format(
                    processed.frame.captured_at.isoformat(),
                    processed.frame.width,
                    processed.frame.height,
                    diagnostics.raw_count,
                    diagnostics.retained_count,
                    diagnostics.capture_duration,
                    diagnostics.inference_duration,
                    diagnostics.processing_duration,
                )
            )
            if isinstance(processed.result, DepthResult):
                record = record + (
                    " depth.raw_minimum={0:.6f} depth.raw_maximum={1:.6f}"
                ).format(
                    processed.result.minimum,
                    processed.result.maximum,
                )
            self._write(record)
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.flush()
        finally:
            if self._owns_stream:
                self._stream.close()

    def __enter__(self) -> "LogOutput":
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> bool:
        if exc_value is None:
            self.close()
            return False
        try:
            self.close()
        except BaseException as error:
            attach_secondary_failure(exc_value, error)
        return False

    def _write(self, record: str) -> None:
        self._stream.write(_redact_text(record) + "\n")
        self._stream.flush()
