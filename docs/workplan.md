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
- Use the smallest supported YOLO model by default and resolve it only from the selected processing type and model-size alias; do not expose an arbitrary model path.
- Default to OpenCV camera index `0`, CPU inference, and a maximum processing rate of 30 frames per second.
- Always pass the input dimensions configured for the resolved model to inference; do not assume one size for every model.
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
- Add the initial CLI options for camera selection, processing type, model selection, inference device, display toggle, and display dimensions; Step 6e replaces their temporary shape with the final grouped contract.
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
- Implement Ultralytics model loading from the configured size alias; Step 6e removes the temporary explicit model-path escape hatch from configuration and CLI.
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

Step 5 is split between completed local preparation and final validation that requires the physical target. Step 6 may be implemented and verified on the laptop while Step 5b waits for the Jetson Nano. Before Step 5b begins, its runbook commands must be aligned with the final Step 6 CLI and configuration files.

### Step 5a: Local deployment preparation — complete

#### Files

- `README.md`
- `docs/jetson-deployment.md`

#### Work

- Document laptop installation and camera use.
- Document the candidate JetPack 4 environment, stock headless container, and the native or custom OpenCV environment required by CSI and display.
- Document the target-side TensorRT export procedure, camera templates, display setup, measurement fields, and acceptance checklist.
- Keep target-only versions and results marked pending rather than claiming untested compatibility.

#### Verification

- Laptop and Jetson instructions use the implemented entry points and options.
- Stock-container and GUI/GStreamer paths are clearly separated.
- TensorRT engines are documented as target-specific and built on the target.
- Hardware-only version, camera, GPU, performance, and thermal results remain explicitly pending.

### Step 5b: Physical Jetson validation — waiting for Jetson Nano

#### Work

- Record the exact board, JetPack/L4T, image digest, Python, OpenCV, PyTorch, TensorRT, Ultralytics, model, and power-mode versions.
- Build the selected nano TensorRT engines on the target device.
- Validate explicit OpenCV USB-camera and configured CSI/GStreamer selection in the final Step 6 application.
- Validate mandatory logging in headless and display runs.
- Measure capture rate, inference latency, end-to-end frame rate, memory, swap, GPU use, power, and thermal behavior under representative UAV settings.
- Record the newest YOLO model and every processing mode demonstrated to work on the target.

#### Verification

- Headless processing captures and processes live frames on the Jetson Nano.
- TensorRT inference uses the Jetson GPU and does not silently fall back to CPU.
- Basic and debug logs contain the required mode-specific information without images or complete result arrays.
- Optional display renders the selected mode at the requested dimensions.
- The application shuts down cleanly and releases camera resources.
- `ruff check .`, `ruff format --check .`, and `pytest` pass in the supported development environment.
- Target-device smoke-test results and measured performance are recorded.

## Step 6: Configuration, processing modes, and mandatory logging

Step 6 replaces temporary silent headless processing with mandatory logging and introduces user-editable defaults plus the remaining requested processing modes. Every substep follows test-first development and leaves the complete suite passing.

### Step 6a: Configuration files and precedence

#### Files

- `pyproject.toml`
- `src/uav_vision/config/defaults/app.json`
- `src/uav_vision/config/defaults/yolo.json`
- `src/uav_vision/config/loader.py`
- `src/uav_vision/config/settings.py`
- configuration tests under `tests/`

#### Work

- Package readable JSON defaults for application settings and mode-specific YOLO settings.
- Put `log_level: basic` in the application defaults rather than relying only on a Python default.
- Add `LogLevel` with `basic` and `debug`, an optional log path, and paths connecting application and YOLO configuration.
- Merge settings with the fixed precedence: explicit CLI values, selected user configuration, then packaged defaults.
- Permit partial user files while rejecting missing files, unknown keys, invalid types, invalid enum values, and invalid mode-specific combinations.
- Resolve a relative YOLO configuration path relative to its application configuration file.
- Mark JSON defaults as package data so installed and editable environments behave the same.

#### Verification

- Packaged defaults alone produce valid settings and prove that `basic` came from the file.
- Partial and complete user overrides preserve unspecified defaults.
- Explicit CLI values win over both file layers.
- Missing files, malformed JSON, unknown keys, wrong types, and invalid values fail descriptively.
- Built and editable installs contain both default files.

### Step 6b: Mode-neutral result, processing, and diagnostics contracts

#### Files

- `src/uav_vision/domain/frame.py`
- `src/uav_vision/domain/detection.py`
- `src/uav_vision/domain/segmentation.py`
- `src/uav_vision/domain/depth.py`
- `src/uav_vision/inference/base.py`
- `src/uav_vision/processing/base.py`
- `src/uav_vision/pipeline.py`
- domain, inference-contract, processing-contract, and pipeline tests under `tests/`

#### Work

- Replace the detection-only `ProcessedFrame` shape with exactly one validated `DetectionResult`, `SegmentationResult`, or `DepthResult`.
- Define separate `Detector`, `Segmenter`, and `DepthEstimator` protocols using application-owned types.
- Validate mask shapes, class metadata, finite depth maps, and agreement with the source frame.
- Define immutable per-frame diagnostics: pipeline-owned capture and processing durations, processor-owned provider inference duration, and mode-owned raw and retained result counts.
- Keep capture, pipeline, and outputs independent from Ultralytics result classes.

#### Verification

- A `ProcessedFrame` without a result, with mixed results, or with a mode-mismatched result cannot be constructed; an empty detection result remains valid.
- Invalid detection fields, masks, class metadata, depth shapes, and non-finite depth values are rejected.
- Diagnostics reject negative or non-finite durations and cannot be mutated after construction.
- The pipeline processes every result variant without importing a provider or display backend.

### Step 6c: Mandatory logger foundation and diagnostics

#### Files

- `src/uav_vision/output/log.py`
- `src/uav_vision/output/__init__.py`
- `src/uav_vision/app.py`
- `src/uav_vision/pipeline.py`
- logger, application, and pipeline tests under `tests/`

#### Work

- Construct the logger immediately after configuration resolution and before camera or model initialization; there is no configuration that disables it.
- Give `LogOutput` explicit startup, diagnostic-event, per-frame, and close operations instead of hiding non-frame logging inside `FrameOutput.write()`.
- At `basic`, report the fully resolved launch configuration exactly once.
- At `debug`, include all basic records plus component initialization, frame identity and dimensions, camera/provider/model/device selection, raw and retained counts, capture/inference/processing timings, and exception tracebacks.
- Write plain-text logs to the console when no path is configured and to the selected file otherwise.
- Open a log file once, flush completed records, close it idempotently, redact credentials embedded in sources, and never log images or complete arrays.
- Keep JSON and JSONL output absent. Do not add rotation, remote transport, or sampling.

#### Verification

- Headless and display runs create the logger before other runtime components and write exactly one launch-configuration record.
- `debug` is an informational superset of `basic`.
- Initialization failures are logged even when camera or model construction does not complete.
- Each timing is measured by its declared owner and appears once in a debug frame record.
- Console and file destinations behave identically and propagate open/write failures.
- Cleanup runs after normal completion, interruption, component failure, and logger failure.

### Step 6d: Detection filtering, reporting, and display

#### Files

- `src/uav_vision/config/defaults/yolo.json`
- `src/uav_vision/config/settings.py`
- `src/uav_vision/processing/detection.py`
- `src/uav_vision/output/log.py`
- `src/uav_vision/output/display.py`
- detection, logger, and display tests under `tests/`

#### Work

- Add detection-only options for selected classes, minimum confidence, and global top-k.
- Apply filters in order: class selection, confidence threshold, stable descending confidence order, then top-k.
- Pass the same filtered `DetectionResult` to logger and display.
- At `basic`, report each frame's class name, confidence, and bounding box for every retained detection, including an explicit empty result.
- Display class name and confidence with each retained bounding box.

#### Verification

- Tests cover no filters, one and many classes, confidence boundaries, stable ties, zero results, and top-k below, equal to, and above the available count.
- Logger and display observe the same ordered detection objects.
- Empty and non-empty detections have deterministic, readable log records.
- Display annotation does not mutate the source frame.

### Step 6e: Grouped CLI and resolved runtime configuration

#### Files

- `src/uav_vision/config/cli.py`
- `src/uav_vision/config/models.py`
- `src/uav_vision/config/defaults/app.json`
- `src/uav_vision/config/defaults/yolo.json`
- `src/uav_vision/config/loader.py`
- `src/uav_vision/config/settings.py`
- `src/uav_vision/app.py`
- `src/uav_vision/output/log.py`
- `src/uav_vision/entrypoints/main.py`
- `src/uav_vision/entrypoints/usb_camera.py`
- CLI, configuration, application, and entry-point tests under `tests/`

#### Work

- Organize CLI help into configuration, camera, processing/inference, display, logging, and runtime groups rather than presenting one flat option list.
- Add `--config-path`, `--log-level {basic,debug}`, `--log-path`, `--fps`, and explicit positive and negative display overrides.
- Keep `--display-width` and `--display-height` together in the display group and apply them only to display configuration; validate both as positive integers.
- Expose `--camera-type {opencv,gstreamer}` and keep `opencv` as the default until a later workplan revision explicitly changes camera selection.
- Replace a numeric `--camera-source` with the accurately named `--camera-index`; accept a non-negative integer and default it to `0` for OpenCV capture. Keep a textual GStreamer pipeline or other non-index source in configuration rather than calling it an index.
- Accept only `cpu` and `cuda` for `--device`, with `cpu` in the packaged defaults. Report unavailable requested CUDA explicitly and never fall back silently.
- Accept `--fps` as a positive integer with packaged default `30`. In this substep it becomes validated immutable configuration; Step 6f applies the limit to the processing loop.
- Parse configuration and processing mode before constructing the final parser so help lists common groups plus only options relevant to `detection`, `segmentation`, or `depth`.
- Remove `--model-path` and the corresponding application setting. Resolve the model exclusively from processing type plus `--model-size`; an Ultralytics identifier may be downloaded or loaded from its normal cache.
- Keep provider-specific identifiers and target TensorRT engine references inside the YOLO configuration and model resolver so another provider can be added without changing CLI contracts.
- Route every entry point through the same configuration loader and model resolver. Do not construct a partial `AppSettings` directly from parser defaults, because that bypasses packaged application and YOLO defaults.
- Complete configuration and model resolution before constructing runtime components or writing the startup record. The selected processing type and model size must always resolve a YOLO origin and concrete model identifier.
- Make the single startup record report the complete effective configuration, including camera type and index or configured source, processing type, model size and resolved model identifier, device, FPS, display enabled state and dimensions, logging level and destination, and YOLO configuration origin. Optional values must be reported as an explicit state such as `disabled` or `console`, not as missing unresolved defaults.
- Do not add a project `models/` directory until the project actually owns local model artifacts.
- Preserve `uav-vision-usb` as a compatibility alias using OpenCV camera index `0` unless explicitly overridden.

#### Verification

- General and mode-specific help expose clearly named groups, with display dimensions in the display group and only the selected mode's processing options.
- CLI values override configuration without treating omitted arguments as overrides.
- Defaults resolve to camera type `opencv`, camera index `0`, device `cpu`, and `fps` `30`.
- Negative or non-integer camera indexes, non-positive or non-integer FPS values, invalid display dimensions, and devices other than `cpu` or `cuda` fail descriptively.
- CLI and settings contain no model-path option; each processing type and model size resolves to one configured Ultralytics identifier or target engine reference.
- A no-argument launch passes through packaged defaults and logs their fully resolved effective values exactly once; required fields such as device, FPS, YOLO origin, and selected model are never logged as `None`.
- Configuration-file and CLI overrides produce the same startup fields with their effective overridden values, while an omitted log path is reported as the `console` destination and disabled display is reported explicitly.
- A requested unavailable CUDA device reports a useful cause without falling back to CPU.
- Existing detection runs remain usable after the CLI migration, including laptop camera index `0` in headless and display modes.

### Step 6f: Model-specific input sizing and frame-rate limiting

#### Files

- `src/uav_vision/config/defaults/yolo.json`
- `src/uav_vision/config/models.py`
- `src/uav_vision/config/settings.py`
- `src/uav_vision/inference/ultralytics.py`
- later mode-specific inference adapters under `src/uav_vision/inference/`
- `src/uav_vision/pipeline.py`
- `src/uav_vision/app.py`
- model-resolution, inference, pipeline, and application tests under `tests/`

#### Work

- Store an explicit inference input size with every processing-type and model-size entry in the YOLO configuration; model resolution returns the model identifier and its input size as one validated selection.
- Extend the startup configuration record with the resolved model input size so the logged model identifier and dimensions always describe the inference call that will run.
- Always pass the resolved input size to Ultralytics for every inference call. Do not rely on one global size or on the provider's implicit default.
- Let the inference adapter perform the provider-supported resize or letterbox operation and return detections, masks, or depth aligned with the application frame contract; display dimensions remain independent from inference dimensions.
- Apply the configured FPS as an upper bound on completed processing iterations using a monotonic clock and an injected wait/clock boundary that can be tested without real delays.
- Define one iteration as inference followed by mandatory logging and, when enabled, display output. Wait only for the unused part of the `1 / fps` period; when work already exceeds the period, continue immediately without overlapping iterations or accumulating delay.
- In headless mode, cap inference plus logging. With display enabled, cap inference plus logging plus display using the same scheduler; do not create separate output rates.
- Keep stop requests, end-of-stream, interruptions, failures, and cleanup responsive and preserve the existing timing ownership used by debug diagnostics.

#### Verification

- Every configured processing-type/model-size pair resolves both a model identifier and a positive supported input size.
- The startup record contains that resolved input size and never reports a different size from the one passed to inference.
- Inference calls receive the selected model's input size for consecutive frames and for models with different configured sizes.
- Results remain aligned with the source frame while optional display independently uses its configured width and height.
- A default run uses 30 FPS, an explicit positive integer overrides it, and invalid values fail during configuration.
- Deterministic clock tests prove that fast iterations wait for the remainder, slow iterations do not wait, and deadlines do not accumulate drift.
- Headless tests include inference and logging inside the capped iteration; display tests additionally include display work inside the same cap.
- Shutdown and cleanup do not wait for an unnecessary next-frame deadline.

### Step 6g: Semantic segmentation

#### Files

- `src/uav_vision/domain/segmentation.py`
- segmentation inference adapter under `src/uav_vision/inference/`
- `src/uav_vision/processing/segmentation.py`
- `src/uav_vision/output/display.py`
- YOLO defaults and segmentation tests under `tests/`

#### Work

- Use the configured nano semantic-segmentation model by default and keep Ultralytics objects inside the adapter.
- Convert semantic masks and class metadata into `SegmentationResult`.
- Log the number of unique classes present in every frame at `basic`; add provider and result diagnostics at `debug`.
- Render a deterministic colour overlay and class labels when display is enabled.

#### Verification

- Tests cover zero, one, and multiple present classes, unknown classes, mismatched mask dimensions, and malformed provider data.
- The logged class count means unique classes present in the frame, not instances or total model classes.
- Headless mode never imports display resources, and display does not mutate source data.

### Step 6h: Monocular depth estimation

#### Files

- `src/uav_vision/domain/depth.py`
- depth inference adapter under `src/uav_vision/inference/`
- `src/uav_vision/processing/depth.py`
- `src/uav_vision/output/display.py`
- YOLO defaults and depth tests under `tests/`

#### Work

- Use the configured nano Ultralytics depth model by default while keeping provider types behind `DepthEstimator`.
- Preserve the model's documented units and scale in `DepthResult` rather than assuming arbitrary normalisation.
- Compute minimum, maximum, mean, and standard deviation once per frame for `basic` logging.
- Add raw-range and timing diagnostics at `debug`.
- Render a controlled colour-map view without modifying source depth values.
- Treat Jetson/TensorRT compatibility as part of pending Step 5b validation.

#### Verification

- Tests cover constant and varying maps, zero range, wrong shapes, non-finite values, statistics, and provider failures.
- Logger statistics match the retained depth map and display normalisation handles degenerate ranges.
- Headless and display paths consume the same `DepthResult`.

### Step 6i: Integration, compatibility, and documentation refresh

#### Files

- `README.md`
- `docs/architecture.md`
- `docs/jetson-deployment.md`
- entry-point and end-to-end tests under `tests/`

#### Work

- Document repository structure, installation profiles, configuration precedence, grouped CLI help, every current flag, all three modes, mandatory logging, model-specific input sizes, FPS limiting, and console/file examples.
- Recheck that the earlier CLI migration removed `--model-path` and numeric `--camera-source` from every example, uses `--camera-index` for OpenCV indexes, and keeps TensorRT references in YOLO configuration.
- Confirm existing capture, inference, cleanup, laptop entry-point, headless, and display behavior remains compatible with the new contracts.
- Run laptop headless and display smoke tests, then update the pending Step 5b checklist with final commands.

#### Verification

- `ruff check .`, `ruff format --check .`, full `pytest`, and `uv lock --check` pass.
- Built and editable installs contain both default configuration files.
- CLI help and option groups match the README for every processing mode.
- There are no JSON/JSONL output references or obsolete public flags.
- Laptop camera index `0` processes frames in headless and display modes with the mandatory logger, CPU default, resolved model input size, and 30 FPS default cap.

## Working rule

For each behavior, first add a focused failing pytest test, run it to confirm the expected failure, implement the smallest code that passes, and rerun the relevant tests followed by the complete suite. Do not begin a later step while an earlier step is failing or unverified.
