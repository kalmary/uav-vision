# UAV Vision

UAV Vision is a real-time camera-processing application built around
Ultralytics YOLO. A run selects one of three modes:

- object detection with class-labelled bounding boxes;
- semantic segmentation with a class-coloured overlay;
- monocular depth estimation.

The capture and inference pipeline is headless by default. Logging is always
enabled, while the OpenCV display is optional.

## Repository structure

```text
.
├── docs/                       Architecture, work plan, and deployment notes
├── models/                     Downloaded weights and local model artifacts
├── src/uav_vision/
│   ├── app.py                  Runtime composition
│   ├── pipeline.py             Capture, processing, output, and FPS loop
│   ├── capture/                OpenCV and GStreamer camera adapters
│   ├── config/                 CLI, settings, loading, and packaged YAML defaults
│   ├── domain/                 Frames and mode-specific result types
│   ├── entrypoints/            `uav-vision` and `uav-vision-usb`
│   ├── inference/              Ultralytics provider adapters
│   ├── output/                 Mandatory logging and optional display
│   └── processing/             Detection, segmentation, and depth processors
├── tests/                      Unit and integration tests
├── pyproject.toml              Package metadata and dependency profiles
└── uv.lock                     Locked dependency resolution
```

## Install

Python 3.8 is the package compatibility floor. Local development uses Python
3.12. Select exactly one runtime profile and include the test dependency group:

Desktop or laptop with GUI-capable OpenCV:

```sh
uv python pin 3.12
uv sync --python 3.12 --extra desktop --group test
```

Generic headless environment:

```sh
uv python pin 3.12
uv sync --python 3.12 --extra headless --group test
```

The `desktop` and `headless` profiles are mutually exclusive because their
OpenCV packages must not coexist. The headless profile cannot open the optional
display window. The project Python pin keeps the commands in the next section
on the installed Python 3.12 environment without repeating profile options.
Jetson-specific installation remains separate; see
[Jetson deployment](docs/jetson-deployment.md).

## Run

Process OpenCV camera index `0` in the default detection mode and write logs to
the console:

```sh
uv run uav-vision
```

`uav-vision-usb` is a compatibility alias with the same options. Packaged
defaults already select OpenCV camera index `0`, and either entry point can
override it explicitly:

```sh
uv run uav-vision-usb --camera-index 1
```

Enable the display on a 1280×720 canvas:

```sh
uv run uav-vision \
    --display --display-width 1280 --display-height 720
```

The display scales each frame uniformly, centres it, and fills unused canvas
space with black padding, so the source aspect ratio is preserved. Detection
draws the filtered boxes and labels, segmentation draws a deterministic colour
overlay and present-class labels, and depth uses a Viridis colour map. Every
view includes measured FPS. Press `q`, `Q`, or `Esc` while the window has focus
to stop.

## Processing modes and models

The packaged default for every mode is the `nano` size. Model identifiers are
kept under `models/`; Ultralytics loads an existing artifact there or obtains
the missing weights on first use.

| Mode | Packaged nano model | Inference input size | Basic per-frame result |
| --- | --- | ---: | --- |
| Detection | `models/yolo26n.pt` | 640 | Retained class name, confidence, and bounding box, or an explicit empty result |
| Semantic segmentation | `models/yolo26n-sem.pt` | 640 | Number of unique classes present in the class map |
| Depth | `models/yolo26n-depth.pt` | 768 | Minimum, maximum, mean, and population standard deviation |

Depth values use metres with scale `1.0`. The other available aliases are
`small`, `medium`, `large`, and `xlarge`; the packaged mappings use input size
640 for detection and segmentation and 768 for depth.

Detection with class IDs `0` or `2`, confidence at least `0.4`, and at most the
five most confident retained detections:

```sh
uv run uav-vision \
    --processing-type detection \
    --selected-classes 0 2 \
    --minimum-confidence 0.4 \
    --top-k 5
```

Detection filters are applied in this order: selected classes, minimum
confidence, stable confidence-descending order, then global top-k. The logger
and display receive the same ordered result.

Semantic segmentation:

```sh
uv run uav-vision --processing-type segmentation
```

Monocular depth estimation with a small model:

```sh
uv run uav-vision \
    --processing-type depth --model-size small
```

## Configuration files

Both configuration files already ship with the application:

- `src/uav_vision/config/defaults/app.yaml` defines capture, mode, model size,
  device, display, logging, FPS, and the YOLO configuration path.
- `src/uav_vision/config/defaults/yolo.yaml` defines the models, inference input
  sizes, and detection filters.

A normal `uv run uav-vision` automatically reads these files. No custom file
needs to be created. The CLI takes its defaults from the configuration files
and overrides only the options explicitly supplied on the command line. The
shipped `app.yaml` points to the packaged `yolo.yaml`.

Configuration is resolved from highest to lowest priority:

1. options explicitly supplied on the command line;
2. a partial application file selected with `--config-path`;
3. the packaged application and YOLO defaults.

For optional custom settings, create a partial application file that overrides
only the values needed for a run, for example `config/uav.yaml`:

```yaml
capture:
  camera_type: opencv
  source: 1
processing:
  processing_type: detection
inference:
  model_size: small
  device: cpu
display:
  enabled: false
  width: 960
  height: 540
log_level: debug
log_path: uav-vision.log
fps: 15
yolo_config_path: custom-yolo.yaml
```

The optional example above selects a separate custom YOLO file. If using that
example, create `config/custom-yolo.yaml` as well; otherwise omit
`yolo_config_path` to keep the shipped YOLO defaults. A relative path is resolved
relative to the application file, not to the working directory. The custom
YOLO file may also be partial:

```yaml
detection:
  small:
    identifier: models/custom-detector.pt
    input_size: 640
detection_filters:
  selected_classes: [0, 2]
  minimum_confidence: 0.4
  top_k: 5
```

Run the configuration and override one value from the CLI:

```sh
uv run uav-vision \
    --config-path config/uav.yaml --fps 20
```

GStreamer uses a textual pipeline in `capture.source`, supplied through the
application configuration. `--camera-index` is only for a non-negative OpenCV
camera index.

Unknown keys, invalid types and values, unavailable requested CUDA, and a model
size missing from the selected mode's YOLO mapping are startup errors.

## Command-line interface

Use the mode-specific help commands: detection adds three options that
do not apply to segmentation or depth:

```sh
uv run uav-vision --processing-type detection --help
uv run uav-vision --processing-type segmentation --help
uv run uav-vision --processing-type depth --help
```

The CLI also accepts `-h`/`--help`. Its remaining options are organized into six
help groups:

| Help group | Option | Meaning |
| --- | --- | --- |
| configuration | `--config-path PATH` | Load a partial application YAML file. |
| camera | `--camera-type {opencv,gstreamer}` | Select the capture adapter. |
| camera | `--camera-index INDEX` | Use a non-negative OpenCV camera index. |
| processing/inference | `--processing-type {detection,segmentation,depth}` | Select one processing mode. |
| processing/inference | `--model-size {nano,small,medium,large,xlarge}` | Select the configured size alias for the current mode. |
| processing/inference | `--device {cpu,cuda}` | Select the inference device. |
| processing/inference | `--selected-classes ID [ID ...]` | Detection only: retain the listed non-negative class IDs. |
| processing/inference | `--minimum-confidence VALUE` | Detection only: set the minimum retained confidence in the inclusive range `0` to `1`. |
| processing/inference | `--top-k COUNT` | Detection only: retain a positive maximum number after sorting. |
| display | `--display` | Enable the processed-video window. |
| display | `--no-display` | Disable the processed-video window. |
| display | `--display-width PIXELS` | Set the positive output-canvas width. |
| display | `--display-height PIXELS` | Set the positive output-canvas height. |
| logging | `--log-level {basic,debug}` | Select result-only or diagnostic logging. |
| logging | `--log-path PATH` | Append logs to a file instead of the console. |
| runtime | `--fps FPS` | Set a non-negative processing-rate cap; `0` is uncapped. |

Packaged defaults are detection, nano, OpenCV camera index `0`, CPU, display
disabled at 1280×720, basic console logging, and a 30 FPS cap. Display dimensions
do not change the model's configured inference input size.

## Logging and frame-rate limiting

Logging cannot be disabled. With no `--log-path`, records go to the console on
standard error. A path switches the same records to an append-only text file:

```sh
uv run uav-vision \
    --log-level debug --log-path uav-vision.log
```

`basic` writes one fully resolved startup record and one mode-specific record
for every processed frame. Every frame record reports measured FPS; the first
uses an explicit unavailable value because there is no earlier frame period.
`debug` includes all basic records plus component initialization,
camera/provider/model/device selection, frame capture timestamp and dimensions,
raw and retained result counts, capture/inference/processing timings, and
failure tracebacks. Logs do not contain image arrays, segmentation maps, or
complete depth maps.

The default `--fps 30` limits complete iterations, including inference,
mandatory logging, and the optional display. `--fps 0` removes the
application-imposed wait and processes at the available hardware rate.

## Verify the local installation

The test group installed with either profile provides the local verification
tools:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

## Ultralytics license

Ultralytics YOLO and its models are offered under AGPL-3.0 or an Enterprise
license. Review the [Ultralytics licensing information](https://docs.ultralytics.com/help/contributing/#license)
before distributing this application or a product that includes it.
