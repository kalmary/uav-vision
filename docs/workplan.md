# Python UAV Vision Work Plan

## Goal

Build the application described in `docs/architecture.md` as a modular Python package. Complete and verify one step before starting the next.

## Initial package selection

Package selection follows `AGENTS.md`: use popular, actively maintained libraries for established problems instead of implementing substitutes. Before implementation, resolve and lock exact versions with `uv` on the development platform, then validate the separate JetPack 4 environment without replacing NVIDIA-provided GPU packages.

### Runtime packages

| Package | Responsibility | Selection reason and constraints |
| --- | --- | --- |
| `ultralytics` | Load YOLO models, run prediction, expose model results, and load or export supported TensorRT engines. | Primary maintained YOLO package with official Jetson support. Use the latest release demonstrated to work in the target JetPack 4 environment. Keep it behind `Detector` so provider-specific types do not escape into application code. Review its AGPL-3.0 or Enterprise licensing requirements before distribution. |
| `numpy` | Store and validate image arrays exchanged between capture, processing, inference, and output. | Standard array representation used directly by OpenCV and Ultralytics. Treat it as a direct dependency even though Ultralytics also requires it. Use the version supplied or supported by the selected Jetson environment. |
| `opencv-python` | Desktop and laptop USB-camera capture, resizing, annotation, and optional window display. | Dominant Python computer-vision package. Install only for desktop environments that need GUI support. |
| `ultralytics-opencv-headless` | Ultralytics inference and OpenCV for non-Jetson environments without graphical display libraries. | Official Ultralytics headless distribution using `opencv-python-headless`. Select it instead of `ultralytics` for a generic headless installation; never install both OpenCV wheel variants in one environment. |

Jetson Nano must use the OpenCV, PyTorch, Torchvision, TensorRT, CUDA, and ONNX Runtime builds supplied by the selected JetPack 4 or official Ultralytics JetPack 4 environment. Do not replace them with generic PyPI wheels. In particular, PyPI OpenCV wheels are CPU-only and may omit the GStreamer integration required for Jetson camera pipelines.

### Installation paths

Running the application requires exactly one platform profile. The dependencies are optional in package metadata only because no single OpenCV and Ultralytics installation is valid for every supported platform.
Python 3.8 is the compatibility floor for the JetPack 4 path; desktop development and the primary local test run use Python 3.12.

- Desktop development: `uv sync --extra desktop --group test`.
- Generic headless development: `uv sync --extra headless --group test`.
- The `desktop` and `headless` extras are mutually exclusive. `uv` rejects selecting both; other installers must be given exactly one extra.
- Jetson Nano: start from the official `ultralytics/ultralytics:latest-jetson-jetpack4` image, verify that its NVIDIA OpenCV, PyTorch, TensorRT, and CUDA imports work, then install only this project with `python3 -m pip install --no-deps -e .`. The Jetson path deliberately has no project extra because resolving generic PyPI dependencies could replace the device-specific packages. Exact target-device versions and smoke-test results remain Step 5 work.
- Optional Jetson display uses the GUI-capable OpenCV supplied by the validated JetPack 4 environment and the same `--display` settings as the laptop path. It must not require the `desktop` extra; Step 5 will document and test the required display forwarding or local desktop setup.

### Development and build packages

| Package | Responsibility | Selection reason and constraints |
| --- | --- | --- |
| `pytest` | Unit and integration test execution. | Established project-standard test framework required by `AGENTS.md`. Select a release compatible with the supported Python floor. |
| `pytest-cov` | Coverage reporting during development. | Widely used pytest integration; measure changed behavior without custom coverage scripts. |
| `ruff` | Formatting and linting. | Maintained single tool replacing separate formatter, import sorter, and basic lint dependencies. Configure it for the supported Python floor. |
| `setuptools` and `wheel` | Build the `src/`-layout Python package. | Ubiquitous maintained packaging backend with editable-install and console-entry-point support. |
| `uv` | Create environments, resolve dependencies, run tools, and maintain the lock file. | Repository-preferred package workflow. It is a development tool rather than an application runtime dependency. |

### Standard-library components

Use `argparse`, `dataclasses`, `enum`, `json`, `pathlib`, `contextlib`, and `typing.Protocol` instead of custom frameworks for CLI parsing, application data, serialization, resource cleanup, and interfaces. These are maintained Python components and keep compatibility with the Python 3.8 runtime used by the official JetPack 4 Ultralytics image. Current Typer and Pydantic releases require newer Python versions, so adding or pinning older framework releases would increase compatibility and maintenance risk without solving a complex project requirement.

Do not initially add a dependency-injection framework, alternate image library, custom CUDA wrapper, GStreamer Python binding, logging framework, or serialization package. Add one only when an implemented requirement cannot be handled clearly by the selected packages or standard library, and record the evidence in this plan and status document.

## Global constraints

- Keep all application code under `src/uav_vision/`; do not add root-level Python scripts.
- Support headless processing without importing or opening display resources.
- Use application-owned domain types across module boundaries.
- Keep camera, inference-provider, processing-mode, and output changes isolated behind small protocols.
- Use the smallest supported YOLO model by default while accepting a model-size alias or explicit model path.
- Report invalid configuration, camera failures, model failures, and processing failures explicitly.
- Develop behavior test-first and run the complete pytest suite after every step.

## Step 1: Python foundation, domain types, and configuration

### Files

- `pyproject.toml`
- `src/uav_vision/__init__.py`
- `src/uav_vision/config/__init__.py`
- `src/uav_vision/config/cli.py`
- `src/uav_vision/config/models.py`
- `src/uav_vision/config/settings.py`
- `src/uav_vision/domain/__init__.py`
- `src/uav_vision/domain/detection.py`
- `src/uav_vision/domain/frame.py`
- configuration and domain tests under `tests/`

### Work

- Configure the package, Python version, runtime dependencies, and development dependencies. Console entry points remain Step 4 work because their application modules do not exist in Step 1.
- Define mutually exclusive desktop and generic-headless Ultralytics installation paths, plus a Jetson installation path that preserves NVIDIA-provided packages.
- Generate the development dependency lock file after the initial environment resolves successfully; do not stage or commit it without explicit approval.
- Define validated frame, bounding-box, detection, and processed-frame data classes.
- Keep the shared frame representation as contiguous BGR `uint8` data.
- Define typed settings for capture, processing, inference, and optional display.
- Add CLI options for camera type and source, processing type, model size or path, inference device, display toggle, and display dimensions.
- Centralize model-size-to-name mappings in `config/models.py`.
- Default to detection, the nano model, headless output, and an appropriate camera source.

### Verification

- Valid arguments and defaults produce the expected settings without opening a camera or loading a model.
- Desktop, headless, and Jetson dependency instructions do not install conflicting OpenCV packages or replace JetPack GPU libraries.
- Invalid dimensions, sources, devices, model selections, and conflicting arguments fail with descriptive messages.
- Domain types reject malformed images, confidence values, and bounding boxes.
- `ruff check .`, `ruff format --check .`, and `pytest` pass.

## Step 2: Replaceable camera capture

### Files

- `src/uav_vision/capture/__init__.py`
- `src/uav_vision/capture/base.py`
- `src/uav_vision/capture/opencv.py`
- `src/uav_vision/capture/gstreamer.py`
- capture tests under `tests/`

### Work

- Define the `FrameSource` protocol with explicit frame, end-of-stream, failure, and cleanup behavior.
- Implement OpenCV capture for numeric USB-camera indexes, video paths, and supported stream URLs.
- Implement Jetson GStreamer capture as a separate adapter using OpenCV's GStreamer backend.
- Assign monotonically increasing sequence numbers and capture timestamps.
- Validate captured image layout before creating an application `Frame`.
- Make cleanup idempotent and usable from a context manager.

### Verification

- Deterministic capture doubles exercise successful frames, end-of-stream, initialization failure, read failure, and cleanup.
- OpenCV and GStreamer construction are tested without requiring physical camera hardware.
- A laptop smoke test confirms that USB camera index `0` produces frames.
- `ruff check .`, `ruff format --check .`, and `pytest` pass.

## Step 3: Replaceable YOLO inference and processing mode

### Files

- `src/uav_vision/inference/__init__.py`
- `src/uav_vision/inference/base.py`
- `src/uav_vision/inference/ultralytics.py`
- `src/uav_vision/processing/__init__.py`
- `src/uav_vision/processing/base.py`
- `src/uav_vision/processing/detection.py`
- inference and processing tests under `tests/`

### Work

- Define the `Detector` protocol in terms of application frames and detections.
- Implement Ultralytics model loading from a configured size alias or explicit model path.
- Pass the selected inference device to Ultralytics without silent fallback in application code.
- Run detection on BGR frames and convert result boxes, class identifiers, labels, and confidence values into application types.
- Define the `FrameProcessor` protocol and implement `DetectionProcessor` as the initial mode.
- Keep Ultralytics result types inside the inference adapter.

### Verification

- Conversion tests cover zero, one, and multiple detections; unknown class identifiers; malformed result data; and provider errors.
- Processor tests use a detector double and return the expected `ProcessedFrame`.
- A known image produces valid labels, confidence values, and image-space bounding boxes using the default nano model.
- `ruff check .`, `ruff format --check .`, and `pytest` pass.

## Step 4: Headless pipeline, outputs, and entry points

### Files

- `src/uav_vision/output/__init__.py`
- `src/uav_vision/output/base.py`
- `src/uav_vision/output/display.py`
- `src/uav_vision/pipeline.py`
- `src/uav_vision/app.py`
- `src/uav_vision/entrypoints/__init__.py`
- `src/uav_vision/entrypoints/main.py`
- `src/uav_vision/entrypoints/usb_camera.py`
- pipeline and entry-point tests under `tests/`

### Work

- Define `FrameOutput` and implement optional display output.
- Implement optional annotation and display without putting windowing code on the headless path.
- Build the processing loop around `FrameSource`, `FrameProcessor`, and zero or more configured outputs.
- Stop cleanly on end-of-stream, display shutdown, keyboard interruption, or an injected shutdown request.
- Propagate component failures and close all initialized resources.
- Compose concrete components in `app` from validated settings.
- Provide the full `uav-vision` entry point.
- Provide `uav-vision-usb`, defaulting to OpenCV camera index `0` for conventional laptops and USB cameras.

### Verification

- Integration tests cover successful headless processing, end-of-stream, shutdown, component errors, and cleanup.
- Display tests verify annotation and requested dimensions without requiring a graphical session.
- Entry-point tests confirm that the USB entry point changes only the camera defaults.
- Manual laptop testing validates live USB capture and optional display.
- `ruff check .`, `ruff format --check .`, and `pytest` pass.

## Step 5: Jetson Nano deployment and validation

### Files

- `README.md`
- deployment configuration or scripts justified by the validated Jetson environment
- Jetson-focused tests or fixtures under `tests/` where hardware is not required

### Work

- Document the supported JetPack 4 Python or container environment and exact dependency versions.
- Export or obtain the nano YOLO model in a TensorRT format supported by the target Jetson Nano.
- Build the engine on the target device when required by TensorRT compatibility.
- Document tested CSI, USB, or GStreamer camera configurations without embedding one board-specific pipeline into the core loop.
- Measure capture rate, inference latency, end-to-end frame rate, memory use, and thermal behavior under representative UAV settings.
- Record the newest YOLO model version demonstrated to work on the target rather than claiming untested compatibility.

### Verification

- Headless processing captures and processes live frames on the Jetson Nano.
- TensorRT inference uses the Jetson GPU and does not silently fall back to CPU.
- Optional display shows labelled bounding boxes at the requested dimensions.
- The application shuts down cleanly and releases camera resources.
- `ruff check .`, `ruff format --check .`, and `pytest` pass in the supported development environment.
- Target-device smoke-test results and measured performance are recorded.

## Working rule

For each behavior, first add a focused failing pytest test, run it to confirm the expected failure, implement the smallest code that passes, and rerun the relevant tests followed by the complete suite. Do not begin a later step while an earlier step is failing or unverified.
