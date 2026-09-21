import re
import sys
import traceback
from typing import Optional, TextIO

from uav_vision.config.settings import AppSettings, LogLevel, LogSettings
from uav_vision.domain import ProcessedFrame


def _redact_source(source: object) -> str:
    if not isinstance(source, str):
        return str(source)
    return re.sub(
        r"([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@",
        r"\1***@",
        source,
    )


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
        self._started = True
        display = (
            "disabled"
            if settings.display is None
            else "{0}x{1}".format(settings.display.width, settings.display.height)
        )
        model_size = (
            "None"
            if settings.inference.model_size is None
            else settings.inference.model_size.value
        )
        model_path = (
            "None"
            if settings.inference.model_path is None
            else str(settings.inference.model_path)
        )
        device = (
            "None" if settings.inference.device is None else settings.inference.device
        )
        path = "None" if self._settings.path is None else str(self._settings.path)
        self._write(
            (
                "startup capture.type={0} capture.source={1} "
                "processing.type={2} inference.model_size={3} "
                "inference.model_path={4} "
                "inference.device={5} display={6} logging.level={7} "
                "logging.path={8}"
            ).format(
                settings.capture.camera_type.value,
                _redact_source(settings.capture.source),
                settings.processing.processing_type.value,
                model_size,
                model_path,
                device,
                display,
                self._settings.level.value,
                path,
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
        if self._settings.level is not LogLevel.DEBUG:
            return
        record = "diagnostic component={0} event={1}".format(
            component, _redact_source(event)
        )
        if error is not None:
            details = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ).rstrip()
            record = record + "\n" + details
        self._write(record)

    def write(self, processed: ProcessedFrame) -> bool:
        if self._closed:
            raise RuntimeError("Log output is closed")
        if self._settings.level is LogLevel.DEBUG:
            diagnostics = processed.diagnostics
            self._write(
                "frame sequence={0} captured_at={1} dimensions={2}x{3} "
                "raw_count={4} retained_count={5} capture_duration={6:.6f} "
                "inference_duration={7:.6f} processing_duration={8:.6f}".format(
                    processed.frame.sequence,
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

    def _write(self, record: str) -> None:
        self._stream.write(record + "\n")
        self._stream.flush()
