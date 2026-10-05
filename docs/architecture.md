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

All application code lives under `src/uav_vision/`. Editable configuration,
project metadata, and documentation live at the repository root; tests live
under `tests/`.

```text
.
├── config/
│   ├── app.yaml
│   └── yolo.yaml
├── models/
├── src/uav_vision/
│   ├── __init__.py
│   ├── app.py
│   ├── pipeline.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── loader.py
│   │   ├── models.py
│   │   └── settings.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── depth.py
│   │   ├── detection.py
│   │   ├── frame.py
│   │   └── segmentation.py
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gstreamer.py
│   │   └── opencv.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ultralytics.py
│   │   └── ultralytics_depth.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── depth.py
│   │   ├── detection.py
│   │   └── segmentation.py
│   ├── output/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── display.py
│   │   └── log.py
│   └── entrypoints/
│       ├── __init__.py
│       ├── main.py
│       └── usb_camera.py
└── tests/
```

This is the implemented source structure. Detection and semantic segmentation share `inference/ultralytics.py`. Tests remain under `tests/`, including end-to-end integration tests with deterministic camera and provider doubles.

## Configuration

The source checkout owns one readily accessible pair of editable YAML defaults:
`config/app.yaml` and `config/yolo.yaml`. Application defaults cover capture,
processing mode, inference, display, logging, FPS, and the path to the YOLO
configuration. YOLO defaults map every supported mode and model size to an
Ultralytics model identifier and input size, and contain detection filters.
PyYAML's safe loader reads configuration as data without constructing arbitrary
Python objects.

For source runs, the loader locates the root `config/` directory from the
project location, independently of the current working directory. A built
installation places the two files under `share/uav-vision/config` and locates
them through the installed distribution's file metadata. This keeps default
lookup independent of the launch directory without duplicating defaults inside
the Python package. Editing a checkout changes runs that use that checkout, not
an unrelated installed copy.

Configuration is resolved in this order, from highest to lowest priority:

1. explicitly provided CLI arguments;
2. values from the file selected by `--config-path`;
3. the shipped application and YOLO defaults.

A user configuration may be partial. Unknown keys, invalid types, invalid enum values, and impossible mode-specific combinations are errors rather than silently ignored values. A relative YOLO configuration path is resolved relative to the application configuration that contains it. The fully resolved settings are validated once and are immutable during processing.

`log_level` is stored in the default application configuration and defaults to `basic`. The CLI exposes the same setting through a `LogLevel` enum with `basic` and `debug` values. An optional log path selects file output; without it, logs go to the console.

## Data model

- `Frame` owns a NumPy image array and capture timestamp.
- `BoundingBox` stores ordered image-space corner coordinates.
- `Detection` stores a class identifier, class name, confidence, and bounding box.
- `DetectionResult` contains the filtered detections for one frame.
- `SegmentationResult` contains a validated, immutable dense integer class map and
  class metadata for every identifier referenced by that map.
- `DepthResult` contains a validated, finite depth map and its summary statistics.
- `ProcessingDiagnostics` contains immutable capture, provider-inference, and processing durations, measured FPS, and mode-specific raw and retained result counts.
- `ProcessedFrame` joins one frame with exactly one result variant matching the selected processing mode.

Using one explicit result variant avoids invalid states such as a frame containing detection, segmentation, and depth results simultaneously. Provider objects are converted to application-owned types before reaching processing outputs.

The common image representation is a contiguous BGR `uint8` array with shape `(height, width, 3)`. OpenCV produces this format directly. An inference adapter converts it when its model requires another representation.

## Extension boundaries

### Camera source

`FrameSource` exposes `read()` and `close()`. `OpenCvCamera` handles USB cameras and conventional streams. `GStreamerCamera` handles device-specific Jetson pipelines without leaking GStreamer configuration into the processing loop.

Both entry points default to OpenCV camera index `0`. The CLI can select a different OpenCV index with `--camera-index` or select `gstreamer` with `--camera-type`; a stream URL, file path, or GStreamer pipeline belongs in the application configuration's `capture.source`. There is no automatic platform detection or camera fallback. A source initialization or read failure is reported explicitly. `uav-vision-usb` uses the same configuration and processing path.

### Inference provider

Separate `Detector`, `Segmenter`, and `DepthEstimator` protocols accept an application `Frame` and return provider-independent values for their mode. Ultralytics adapters own model loading, device selection, inference, and result conversion.

The semantic adapter converts provider output into a source-sized,
application-owned `SegmentationResult`. It copies the dense class map and includes
the names of classes present in that map in class-identifier order; provider result
objects do not cross the adapter boundary.

The depth adapter uses the Ultralytics `depth` task and copies its source-sized
floating-point map into `DepthResult`. YOLO26 depth predictions are in metres,
so the adapter retains raw values with unit `metre` and scale `1.0`. Default
depth models use input size `768`, with `models/yolo26n-depth.pt` as the default.
`DepthResult` computes its minimum, maximum, mean, and population standard
deviation once from the retained immutable map. Statistics describe raw values;
the unit and scale are reported alongside them.

Model-size aliases and mode-specific model mappings come from the YOLO configuration; each size mapping atomically owns its model identifier and input size. The public CLI selects a supported size rather than an arbitrary model path. Default identifiers resolve under the project `models/` directory so Ultralytics loads existing weights there and downloads missing weights there. A target-specific TensorRT engine can be configured for Jetson without exposing provider details elsewhere.

### Processing mode

`FrameProcessor` accepts a frame and returns a `ProcessedFrame`. Detection, semantic segmentation, and depth each have a processor and matching inference protocol. A mode factory selects the pair explicitly without changing camera capture or the pipeline loop.

Detection filtering is applied before outputs in a fixed order: selected classes, minimum confidence, stable confidence-descending ordering, then global top-k. Logger and display therefore observe the same detections. Mode-aware CLI help exposes only options valid for the selected processing type.

### Output and logging

`FrameOutput` consumes a `ProcessedFrame` and may request shutdown. `LogOutput` is always present in headless and display runs. `DisplayOutput` is an optional second output and owns all windowing behavior. It scales annotated frames uniformly into the configured dimensions and centers them on a black canvas, preserving their aspect ratio without changing frames used by headless processing.

For semantic segmentation, display assigns each non-negative class identifier a
deterministic BGR colour, blends the dense colour map over a copy of the source
frame at 40% colour and 60% source image, and displays only FPS text, without
class labels. Present class IDs, names, and basic colour names are logged in
class-identifier order using the same palette as the display's BGR overlay.
Colours are converted to RGB and matched by squared Euclidean distance to ten
references: black, white, red, orange, yellow, green, blue, purple, pink, and
brown. Equal distances use the first reference in that order.
The source frame and segmentation result remain unchanged.
Headless composition neither constructs `DisplayOutput` nor imports its OpenCV
windowing resources.

Depth display normalizes a temporary copy of the map between its retained minimum
and maximum, then applies the Viridis colour map. Constant maps use the lowest
colour-map value. The resulting view uses the same aspect-ratio-preserving
padding and final-canvas FPS header as segmentation; raw depth values and source
frames remain unchanged. Debug depth logs include the raw range and the shared
capture, inference, and processing timings. Jetson/TensorRT validation remains
pending in Step 5b.

At `basic`, the logger writes the effective launch configuration once and one filtered summary per processed frame:

- detection: class name, confidence, and bounding box for every retained detection, including an explicit empty result;
- semantic segmentation: the number of unique class identifiers present in the
  dense map, rather than the number of instances or available metadata entries,
  plus the present class IDs, names, and nearest basic colour names;
- depth: minimum, maximum, mean, and standard deviation.

Every per-frame record includes measured FPS. The first processed frame reports
an explicit unavailable state because no preceding frame period exists. Display
output overlays the same measured value when enabled.

At `debug`, the logger includes everything from `basic` plus capture timestamps and frame dimensions, camera/provider/model/device selection, raw and retained result counts, component initialization details, capture/provider-inference/processing timings, and tracebacks for failures. The pipeline owns capture and total processing measurements; the selected processor owns provider-inference measurement. The logger consumes these diagnostics and does not attempt to infer timings from output order.

`LogOutput` has explicit operations for the startup configuration, diagnostic events, per-frame results, and cleanup. It is constructed immediately after configuration resolution so camera and model initialization failures are also recorded. Output consists of text records, not images, masks, or complete depth maps. A log file is opened once in append mode and closed with other application resources; rotation and remote transport remain out of scope.

## Entry points

- `uav-vision` exposes the complete CLI with OpenCV camera index `0` as its shipped default.
- `uav-vision-usb` is a laptop-friendly alias with the same defaults and model, display, processing, configuration, and logging options.

Both entry points call the same application composition and processing pipeline.

## Data flow

1. An entry point reads the configuration path and processing mode needed to build mode-aware CLI help.
2. The loader merges the shipped defaults, the selected configuration file, and explicit CLI values, then validates one effective configuration.
3. `app` constructs the logger and reports the effective launch configuration once.
4. `app` selects a camera source and constructs the mode-specific inference adapter and processor, reporting initialization diagnostics through the logger.
5. `pipeline` reads one frame, measures capture, and passes the frame to the processor.
6. The processor measures provider inference, converts the provider result, and applies mode-specific filtering.
7. The pipeline attaches the completed diagnostics to the application-owned result.
8. The mandatory logger reports the mode-specific result; optional display renders the same result.
9. Processing stops on end-of-stream, a user request, or an unrecoverable error.
10. The application closes the camera, logger, and optional display resources exactly once.

## Platform strategy

- Laptop development uses a standard USB camera through OpenCV. The application supports CPU inference by default and explicit CUDA selection when available; other provider devices are not exposed by the configuration contract.
- Jetson Nano uses JetPack 4 and may use an OpenCV GStreamer pipeline or USB camera.
- Jetson deployment uses an Ultralytics-compatible TensorRT engine built and validated on the target device.
- The application does not silently replace an explicitly selected inference device or model.
- Missing requested CUDA produces a concise configuration error; debug logging retains the underlying diagnostics.

## Verification

- `ruff check .` validates Python code quality.
- `ruff format --check .` validates formatting.
- `pytest` validates configuration precedence, mode-aware CLI behavior, adapters, filtering, logging, result conversion, pipeline behavior, and resource cleanup.
- Laptop smoke testing validates USB-camera capture, mandatory logging, and optional display.
- Jetson Nano smoke testing remains deferred and covers configured camera capture, TensorRT model loading, mandatory logging, headless operation, performance, and optional display.
