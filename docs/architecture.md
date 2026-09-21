# UAV Vision Architecture

## Goal

Run real-time object detection in Python on an NVIDIA Jetson Nano or a conventional laptop. The application captures camera frames, runs an Ultralytics YOLO model, produces class-labelled bounding boxes for every processed frame, and optionally displays annotated video.

The capture and inference pipeline must remain independent from display code so it can run headlessly on the UAV.

## Design principles

- Keep application-owned frames, detections, and configuration independent from OpenCV and Ultralytics types.
- Put camera, model-provider, processing-mode, and output variations behind small protocols.
- Select concrete implementations in one composition layer.
- Prefer plain data classes and direct control flow over inheritance hierarchies or general-purpose registries.
- Add a new option by implementing one existing protocol and adding one explicit selection entry.

## Project structure

All application code lives under `src/uav_vision/`. The repository root contains only project metadata and documentation; tests live under `tests/`.

```text
src/uav_vision/
├── __init__.py
├── app.py
├── pipeline.py
├── config/
│   ├── __init__.py
│   ├── cli.py
│   ├── models.py
│   └── settings.py
├── domain/
│   ├── __init__.py
│   ├── detection.py
│   └── frame.py
├── capture/
│   ├── __init__.py
│   ├── base.py
│   ├── gstreamer.py
│   └── opencv.py
├── inference/
│   ├── __init__.py
│   ├── base.py
│   └── ultralytics.py
├── processing/
│   ├── __init__.py
│   ├── base.py
│   └── detection.py
├── output/
│   ├── __init__.py
│   ├── base.py
│   └── display.py
└── entrypoints/
    ├── __init__.py
    ├── main.py
    └── usb_camera.py
```

Tests live under `tests/`, separated into focused unit tests and end-to-end integration tests with deterministic test doubles.

## Data model

- `Frame` owns a NumPy image array, sequence number, and capture timestamp.
- `BoundingBox` stores ordered image-space corner coordinates.
- `Detection` stores a class identifier, class name, confidence, and bounding box.
- `ProcessedFrame` joins one frame with the processing results produced for it.

The common in-memory image representation is a contiguous BGR `uint8` array with shape `(height, width, 3)`. OpenCV produces this format directly. An inference adapter is responsible for converting it if its model provider requires another representation.

## Extension boundaries

### Camera source

`FrameSource` exposes `read()` and `close()`. `OpenCvCamera` handles numeric USB-camera indexes, video paths, and supported stream URLs. `GStreamerCamera` handles device-specific Jetson pipelines without leaking GStreamer configuration into the processing loop.

Adding another camera requires a new `FrameSource` implementation and one configuration selection entry.

### Inference provider

`Detector` accepts an application `Frame` and returns application `Detection` values. `UltralyticsDetector` owns model loading, device selection, inference, and conversion from Ultralytics results.

Model-size aliases and their corresponding model names live in `config/models.py`. An explicit model path bypasses the alias mapping. This keeps future YOLO model-name changes in one place.

### Processing mode

`FrameProcessor` accepts a frame and returns a `ProcessedFrame`. `DetectionProcessor` delegates detection to a `Detector`. Future segmentation, pose, or tracking modes can add processors and domain result types without changing camera capture or the pipeline loop.

### Output

`FrameOutput` consumes a `ProcessedFrame` and may request shutdown. `DisplayOutput` draws labels and boxes and owns all windowing behavior. The pipeline also accepts no outputs, which keeps capture and inference independent from display. Filtered result logging will be added as a separate output.

## Entry points

- `uav-vision` exposes the complete CLI and supports OpenCV or GStreamer capture.
- `uav-vision-usb` is a laptop-friendly entry point that defaults to USB camera index `0` while retaining the same model, display, and processing options.

Both entry points call the same application composition and processing pipeline.

## Data flow

1. An entry point parses and validates CLI arguments.
2. `app` constructs the selected frame source, detector, processor, and outputs.
3. `pipeline` reads one frame and passes it to the processor.
4. The processor returns application-owned results.
5. Configured outputs publish results and optionally display the annotated frame.
6. Processing stops on end-of-stream, a user request, or an unrecoverable error.
7. The application closes the camera and display resources exactly once.

## Platform strategy

- Laptop development uses a standard USB camera through OpenCV and may run inference on CPU, CUDA, Apple MPS, or another device supported by the installed Ultralytics package.
- Jetson Nano uses JetPack 4 and may use an OpenCV GStreamer pipeline for camera capture.
- Jetson deployment should use the Ultralytics JetPack 4 environment and a TensorRT engine built and validated on the target device.
- The application does not silently replace an explicitly selected inference device or model with another option.

## Verification

- `ruff check .` validates Python code quality.
- `ruff format --check .` validates formatting.
- `pytest` validates configuration, adapters, inference-result conversion, pipeline behavior, and resource cleanup.
- Laptop smoke testing validates USB-camera capture and optional display.
- Jetson Nano smoke testing validates GStreamer capture, TensorRT model loading, headless operation, performance, and optional display.
