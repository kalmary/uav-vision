import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from uav_vision.config.settings import ProcessingType
from uav_vision.domain import (
    DepthResult,
    DetectionResult,
    Frame,
    ProcessedFrame,
    ProcessingDiagnostics,
    SegmentationClass,
    SegmentationResult,
)
from uav_vision.pipeline import run_pipeline


def frame():
    return Frame(
        image=np.zeros((12, 16, 3), dtype=np.uint8),
        captured_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )


class SourceDouble:
    def __init__(self, frames=(), error=None):
        self._frames = iter(frames)
        self._error = error
        self.read_calls = 0
        self.close_calls = 0

    def read(self):
        self.read_calls += 1
        if self._error is not None:
            raise self._error
        return next(self._frames, None)

    def close(self):
        self.close_calls += 1


class ProcessorDouble:
    def __init__(self, error=None, results=None):
        self._error = error
        self._results = results
        self.calls = []

    def process(self, input_frame):
        self.calls.append(input_frame)
        if self._error is not None:
            raise self._error
        if self._results is not None:
            return self._results[len(self.calls) - 1]
        return ProcessedFrame(
            input_frame,
            DetectionResult(()),
            ProcessingDiagnostics(None, 0.0, None, 0, 0),
        )


class OutputDouble:
    def __init__(self, should_stop=False, error=None, events=None, name=None):
        self._should_stop = should_stop
        self._error = error
        self._events = events
        self._name = name
        self.calls = []
        self.close_calls = 0

    def write(self, processed):
        self.calls.append(processed)
        if self._events is not None:
            self._events.append(self._name)
        if self._error is not None:
            raise self._error
        return self._should_stop

    def close(self):
        self.close_calls += 1


def test_stops_before_reading_when_shutdown_callback_immediately_requests_stop():
    source = SourceDouble((frame(),))
    processor = ProcessorDouble()

    run_pipeline(source, processor, (), should_stop=lambda: True)

    assert source.read_calls == 0
    assert processor.calls == []


def test_stops_when_source_reaches_end_of_stream():
    source = SourceDouble()
    processor = ProcessorDouble()

    run_pipeline(source, processor, ())

    assert source.read_calls == 1
    assert processor.calls == []


def test_processes_each_frame_once_before_end_of_stream():
    first = frame()
    second = frame()
    source = SourceDouble((first, second))
    processor = ProcessorDouble()
    output = OutputDouble()

    run_pipeline(source, processor, (output,))

    assert source.read_calls == 3
    assert processor.calls == [first, second]
    assert [processed.frame for processed in output.calls] == [first, second]


def test_delivers_the_same_processed_frame_to_all_outputs_before_stopping():
    input_frame = frame()
    source = SourceDouble((input_frame,))
    processed = ProcessedFrame(
        input_frame,
        DetectionResult(()),
        ProcessingDiagnostics(None, 0.0, None, 0, 0),
    )
    processor = ProcessorDouble(results=(processed,))
    events = []
    first_output = OutputDouble(True, events=events, name="first")
    second_output = OutputDouble(events=events, name="second")

    run_pipeline(source, processor, (first_output, second_output))

    assert first_output.calls[0] is second_output.calls[0]
    assert first_output.calls[0].frame is processed.frame
    assert first_output.calls[0].result is processed.result
    assert events == ["first", "second"]


def test_does_not_read_another_frame_after_an_output_requests_stop():
    first = frame()
    source = SourceDouble((first, frame()))
    processor = ProcessorDouble()
    output = OutputDouble(True)

    run_pipeline(source, processor, (output,))

    assert source.read_calls == 1
    assert processor.calls == [first]


def test_checks_shutdown_callback_before_each_read_boundary():
    source = SourceDouble((frame(), frame()))
    processor = ProcessorDouble()
    read_counts = []

    def should_stop():
        read_counts.append(source.read_calls)
        return False

    run_pipeline(source, processor, (), should_stop=should_stop)

    assert read_counts == [0, 1, 2]


def test_processes_frames_without_outputs():
    source = SourceDouble((frame(),))
    processor = ProcessorDouble()

    run_pipeline(source, processor, ())

    assert len(processor.calls) == 1


def test_propagates_shutdown_callback_errors_unchanged():
    error = RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError) as raised:
        run_pipeline(
            SourceDouble(),
            ProcessorDouble(),
            (),
            should_stop=lambda: _raise(error),
        )

    assert raised.value is error


def test_propagates_source_errors_unchanged():
    error = RuntimeError("read failed")

    with pytest.raises(RuntimeError) as raised:
        run_pipeline(SourceDouble(error=error), ProcessorDouble(), ())

    assert raised.value is error


def test_propagates_processor_errors_unchanged():
    error = RuntimeError("processing failed")

    with pytest.raises(RuntimeError) as raised:
        run_pipeline(SourceDouble((frame(),)), ProcessorDouble(error=error), ())

    assert raised.value is error


@pytest.mark.parametrize("index", (0, 1))
def test_propagates_output_errors_unchanged(index):
    error = RuntimeError("output failed")
    outputs = [OutputDouble(), OutputDouble()]
    outputs[index] = OutputDouble(error=error)

    with pytest.raises(RuntimeError) as raised:
        run_pipeline(SourceDouble((frame(),)), ProcessorDouble(), outputs)

    assert raised.value is error


def test_propagates_keyboard_interrupt_unchanged():
    error = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt) as raised:
        run_pipeline(SourceDouble(error=error), ProcessorDouble(), ())

    assert raised.value is error


def test_leaves_source_and_outputs_open_for_the_application_to_close():
    source = SourceDouble((frame(),))
    processor = ProcessorDouble()
    output = OutputDouble()

    run_pipeline(source, processor, (output,))

    assert source.close_calls == 0
    assert output.close_calls == 0


def test_pipeline_attaches_its_capture_and_processing_measurements():
    input_frame = frame()
    processed = ProcessedFrame(
        input_frame,
        DetectionResult(()),
        ProcessingDiagnostics(None, 0.25, None, 0, 0),
    )
    output = OutputDouble()

    run_pipeline(
        SourceDouble((input_frame,)),
        ProcessorDouble(results=(processed,)),
        (output,),
        clock=iter((1.0, 1.05, 2.0, 2.3, 3.0, 3.1)).__next__,
    )

    diagnostics = output.calls[0].diagnostics
    assert diagnostics.capture_duration == pytest.approx(0.05)
    assert diagnostics.inference_duration == 0.25
    assert diagnostics.processing_duration == pytest.approx(0.3)


def test_pipeline_limits_fast_iterations_without_accumulating_deadlines():
    first = frame()
    second = frame()
    output = OutputDouble()
    waits = []

    run_pipeline(
        SourceDouble((first, second)),
        ProcessorDouble(),
        (output,),
        fps=10,
        clock=iter(
            (
                0.0,
                0.01,
                0.02,
                0.03,
                0.04,
                0.1,
                0.11,
                0.12,
                0.13,
                0.14,
                0.2,
            )
        ).__next__,
        wait=waits.append,
    )

    assert waits == pytest.approx([0.06, 0.06])
    assert output.calls[0].diagnostics.frames_per_second is None
    assert output.calls[1].diagnostics.frames_per_second == pytest.approx(10.0)


def test_pipeline_does_not_wait_when_processing_exceeds_the_target_period():
    waits = []

    run_pipeline(
        SourceDouble((frame(),)),
        ProcessorDouble(),
        (OutputDouble(),),
        fps=10,
        clock=iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5)).__next__,
        wait=waits.append,
    )

    assert waits == []


def test_pipeline_zero_fps_runs_without_waiting():
    waits = []

    run_pipeline(
        SourceDouble((frame(),)),
        ProcessorDouble(),
        (OutputDouble(),),
        fps=0,
        clock=iter((0.0, 0.01, 0.02, 0.03, 0.04, 0.05)).__next__,
        wait=waits.append,
    )

    assert waits == []


def test_pipeline_does_not_wait_after_an_output_requests_stop():
    waits = []

    run_pipeline(
        SourceDouble((frame(),)),
        ProcessorDouble(),
        (OutputDouble(should_stop=True),),
        fps=10,
        clock=iter((0.0, 0.01, 0.02, 0.03)).__next__,
        wait=waits.append,
    )

    assert waits == []


def test_pipeline_interrupts_fps_wait_when_shutdown_is_requested():
    current_time = [0.0]
    shutdown_requested = [False]
    waits = []
    source = SourceDouble((frame(), frame()))

    def clock():
        return current_time[0]

    def wait(duration):
        waits.append(duration)
        current_time[0] += duration
        shutdown_requested[0] = True

    run_pipeline(
        source,
        ProcessorDouble(),
        (OutputDouble(),),
        should_stop=lambda: shutdown_requested[0],
        fps=10,
        clock=clock,
        wait=wait,
    )

    assert waits == pytest.approx([0.05])
    assert sum(waits) < 0.1
    assert source.read_calls == 1


@pytest.mark.parametrize(
    "processing_type,result",
    [
        (
            ProcessingType.SEGMENTATION,
            SegmentationResult(
                np.zeros((12, 16), dtype=np.uint8),
                (SegmentationClass(0, "background"),),
            ),
        ),
        (
            ProcessingType.DEPTH,
            DepthResult(np.ones((12, 16), dtype=np.float32), "metre", 0.001),
        ),
    ],
)
def test_pipeline_delivers_each_non_detection_result_variant(processing_type, result):
    input_frame = frame()
    processed = ProcessedFrame(
        input_frame,
        result,
        ProcessingDiagnostics(None, 0.0, None, 1, 1),
        processing_type,
    )
    output = OutputDouble()

    run_pipeline(
        SourceDouble((input_frame,)), ProcessorDouble(results=(processed,)), (output,)
    )

    assert output.calls[0].result is result


def test_importing_pipeline_does_not_import_concrete_runtime_components():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import uav_vision.pipeline; "
            "forbidden = {'uav_vision.capture.opencv', "
            "'uav_vision.capture.gstreamer', 'uav_vision.processing.detection', "
            "'uav_vision.output.display', 'uav_vision.inference.ultralytics'}; "
            "assert not forbidden.intersection(sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _raise(error):
    raise error
