# UAV Vision Architecture

## Goal

Run real-time vision processing in Python on an NVIDIA Jetson Nano or a conventional laptop. The application captures camera frames and runs one selected Ultralytics mode: object detection with class-labelled bounding boxes, semantic segmentation, or monocular depth estimation.

Every run reports its effective configuration and mode-specific results through a logger. Display remains optional, and the capture and inference pipeline remains independent from windowing code so it can run headlessly on the UAV.

## Design principles

- Keep application-owned frames, results, and configuration independent from OpenCV and Ultralytics types.
- Put camera, model-provider, processing-mode, and output variations behind small protocols.
- Select concrete implementations in one composition layer.
- Resolve configuration once at startup with explicit precedence and keep the effective settings immutable.
- Send the same filtered processing result to the mandatory logger and optional display.
- Prefer plain data classes and direct control flow over inheritance hierarchies or general-purpose registries.
- Add a new option by implementing one existing protocol and adding one explicit selection entry.

## Project structure

All application code lives under `src/uav_vision/`. The repository root contains project metadata and documentation; tests live under `tests/`.

```text
src/uav_vision/
├── __init__.py
├── app.py
├── pipeline.py
├── config/
│   ├── __init__.py
│   ├── cli.py
│   ├── loader.py
│   ├── models.py
│   ├── settings.py
│   └── defaults/
│       ├── app.json
│       └── yolo.json
├── domain/
│   ├── __init__.py
│   ├── depth.py
│   ├── detection.py
│   ├── frame.py
│   └── segmentation.py
├── capture/
│   ├── __init__.py
│   ├── auto.py
│   ├── base.py
│   ├── gstreamer.py
│   └── opencv.py
├── inference/
│   ├── __init__.py
│   ├── base.py
│   ├── ultralytics.py
│   ├── ultralytics_depth.py
│   └── ultralytics_segmentation.py
├── processing/
│   ├── __init__.py
│   ├── base.py
│   ├── depth.py
│   ├── detection.py
│   └── segmentation.py
├── output/
│   ├── __init__.py
│   ├── base.py
│   ├── display.py
│   └── log.py
└── entrypoints/
    ├── __init__.py
    ├── main.py
    └── usb_camera.py
```

The additional files shown above are the target structure for Step 6; they are not all implemented yet. Tests remain under `tests/`, separated into focused unit tests and end-to-end integration tests with deterministic doubles.

## Configuration

The package supplies editable JSON defaults in `config/defaults/app.json` and `config/defaults/yolo.json`. Application defaults cover capture, processing mode, display, logging, and the path to the YOLO configuration. YOLO defaults map every supported mode and model size to an Ultralytics model identifier or a validated target-specific engine.

Configuration is resolved in this order, from highest to lowest priority:

1. explicitly provided CLI arguments;
2. values from the file selected by `--config-path`;
3. packaged default files.

A user configuration may be partial. Unknown keys, invalid types, invalid enum values, and impossible mode-specific combinations are errors rather than silently ignored values. A relative YOLO configuration path is resolved relative to the application configuration that contains it. The fully resolved settings are validated once and are immutable during processing.

`log_level` is stored in the default application configuration and defaults to `basic`. The CLI exposes the same setting through a `LogLevel` enum with `basic` and `debug` values. An optional log path selects file output; without it, logs go to the console.

## Data model

- `Frame` owns a NumPy image array, sequence number, and capture timestamp.
- `BoundingBox` stores ordered image-space corner coordinates.
- `Detection` stores a class identifier, class name, confidence, and bounding box.
- `DetectionResult` contains the filtered detections for one frame.
- `SegmentationResult` contains validated semantic class masks and class metadata.
- `DepthResult` contains a validated, finite depth map and its summary statistics.
- `ProcessingDiagnostics` contains immutable capture, provider-inference, and processing durations plus mode-specific raw and retained result counts.
- `ProcessedFrame` joins one frame with exactly one result variant matching the selected processing mode.

Using one explicit result variant avoids invalid states such as a frame containing detection, segmentation, and depth results simultaneously. Provider objects are converted to application-owned types before reaching processing outputs.

The common image representation is a contiguous BGR `uint8` array with shape `(height, width, 3)`. OpenCV produces this format directly. An inference adapter converts it when its model requires another representation.

## Extension boundaries

### Camera source

`FrameSource` exposes `read()` and `close()`. `OpenCvCamera` handles USB cameras and conventional streams. `GStreamerCamera` handles device-specific Jetson pipelines without leaking GStreamer configuration into the processing loop.

Normal CLI use does not require a camera type or source. Platform-aware selection defaults to OpenCV camera `0` on a laptop and tries configured Jetson candidates in deterministic order. Advanced source overrides remain configuration values. Failed candidates are reported together; debug logging records every attempt and basic logging records the selected source. `uav-vision-usb` remains a compatibility alias.

### Inference provider

Separate `Detector`, `Segmenter`, and `DepthEstimator` protocols accept an application `Frame` and return provider-independent values for their mode. Ultralytics adapters own model loading, device selection, inference, and result conversion.

Model-size aliases and mode-specific model mappings come from the YOLO configuration. The public CLI selects a supported size rather than an arbitrary model path. A target-specific TensorRT engine can be configured for Jetson without exposing provider details elsewhere. A local `models/` directory is added only if the project starts owning model files; downloaded Ultralytics models do not justify it by themselves.

### Processing mode

`FrameProcessor` accepts a frame and returns a `ProcessedFrame`. Detection, semantic segmentation, and depth each have a processor and matching inference protocol. A mode factory selects the pair explicitly without changing camera capture or the pipeline loop.

Detection filtering is applied before outputs in a fixed order: selected classes, minimum confidence, stable confidence-descending ordering, then global top-k. Logger and display therefore observe the same detections. Mode-aware CLI help exposes only options valid for the selected processing type.

### Output and logging

`FrameOutput` consumes a `ProcessedFrame` and may request shutdown. `LogOutput` is always present in headless and display runs. `DisplayOutput` is an optional second output and owns all windowing behavior.

At `basic`, the logger writes the effective launch configuration once and one filtered summary per processed frame:

- detection: class name, confidence, and bounding box for every retained detection, including an explicit empty result;
- semantic segmentation: the number of unique classes present in the frame;
- depth: minimum, maximum, mean, and standard deviation.

At `debug`, the logger includes everything from `basic` plus frame identity and dimensions, camera/provider/model/device selection, raw and retained result counts, component initialization details, capture/provider-inference/processing timings, and tracebacks for failures. The pipeline owns capture and total processing measurements; the selected processor owns provider-inference measurement. The logger consumes these diagnostics and does not attempt to infer timings from output order.

`LogOutput` has explicit operations for the startup configuration, diagnostic events, per-frame results, and cleanup. It is constructed immediately after configuration resolution so camera and model initialization failures are also recorded. It does not write images, masks, complete depth maps, JSON, or JSONL. A log file is opened once and closed with other application resources; rotation and remote transport remain out of scope.

## Entry points

- `uav-vision` exposes the complete CLI and selects the normal camera source for the current platform.
- `uav-vision-usb` remains a laptop-friendly compatibility entry point that selects USB camera index `0` while retaining the same model, display, processing, configuration, and logging options.

Both entry points call the same application composition and processing pipeline.

## Data flow

1. An entry point reads the configuration path and processing mode needed to build mode-aware CLI help.
2. The loader merges packaged defaults, the selected configuration file, and explicit CLI values, then validates one effective configuration.
3. `app` constructs the logger and reports the effective launch configuration once.
4. `app` selects a camera source and constructs the mode-specific inference adapter and processor, reporting initialization diagnostics through the logger.
5. `pipeline` reads one frame, measures capture, and passes the frame to the processor.
6. The processor measures provider inference, converts the provider result, and applies mode-specific filtering.
7. The pipeline attaches the completed diagnostics to the application-owned result.
8. The mandatory logger reports the mode-specific result; optional display renders the same result.
9. Processing stops on end-of-stream, a user request, or an unrecoverable error.
10. The application closes the camera, logger, and optional display resources exactly once.

## Platform strategy

- Laptop development uses a standard USB camera through OpenCV and may run inference on CPU, CUDA, Apple MPS, or another device supported by the installed Ultralytics package.
- Jetson Nano uses JetPack 4 and may use an OpenCV GStreamer pipeline or USB camera.
- Jetson deployment uses an Ultralytics-compatible TensorRT engine built and validated on the target device.
- The application does not silently replace an explicitly selected inference device or model.
- Missing requested CUDA produces a concise configuration error; debug logging retains the underlying diagnostics.

## Verification

- `ruff check .` validates Python code quality.
- `ruff format --check .` validates formatting.
- `pytest` validates configuration precedence, mode-aware CLI behavior, adapters, filtering, logging, result conversion, pipeline behavior, and resource cleanup.
- Laptop smoke testing validates USB-camera capture, mandatory logging, and optional display.
- Jetson Nano smoke testing validates automatic camera selection, TensorRT model loading, mandatory logging, headless operation, performance, and optional display.
