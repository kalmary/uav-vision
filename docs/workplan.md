# UAV Vision Work Plan

Implement the application in five independently testable steps. Complete and verify each step before starting the next one.

## Step 1: Domain Types and CLI Configuration

### Files

- `src/domain.rs`
- `src/domain/frame.rs`
- `src/domain/detection.rs`
- `src/config.rs`
- `src/config/cli.rs`

### Work

- Define the frame and pixel-format representations.
- Define bounding-box and detection representations.
- Add CLI options for the camera, processing type, model size or path, inference device, display toggle, and display dimensions.
- Validate missing, invalid, and incompatible configuration values.
- Write unit tests in the corresponding source files before implementing each behavior.

### Completion criteria

- Configuration can be parsed without opening a camera or loading a model.
- Valid arguments produce the expected settings.
- Invalid arguments produce descriptive errors.
- `cargo fmt --check` and `cargo test` pass.

## Step 2: Camera Capture

### Files

- `src/capture.rs`
- `src/capture/camera.rs`

### Work

- Define the common frame-source interface used by the pipeline.
- Open the camera connected to the Jetson Nano.
- Convert captured images into application-owned frames.
- Attach frame sequence numbers, capture times, dimensions, and pixel-format information.
- Report camera initialization, read, and end-of-stream conditions explicitly.
- Unit-test configuration and frame-conversion logic without requiring physical camera hardware.

### Completion criteria

- The application can capture and inspect frames without running YOLO.
- Capture failures are returned to the caller.
- `cargo fmt --check` and `cargo test` pass.

## Step 3: YOLO Inference

### Files

- `src/inference.rs`
- `src/inference/yolo.rs`

### Work

- Define the object-detection boundary used by the pipeline.
- Load an Ultralytics ONNX model through `ultralytics-inference`.
- Translate CLI model-size selection into a supported model name or explicit model path.
- Configure CPU, CUDA, or TensorRT execution.
- Run inference on one application frame.
- Convert Ultralytics results into application-owned detections.
- Test result conversion and configuration without depending on Jetson hardware.

### Completion criteria

- A known image produces class identifiers, labels, confidence values, and valid bounding boxes.
- Model-loading and inference errors are returned to the caller.
- CPU inference works before Jetson acceleration is enabled.
- `cargo fmt --check` and `cargo test` pass.

## Step 4: Headless Processing Pipeline

### Files

- `src/pipeline.rs`
- `src/app.rs`
- `src/lib.rs`
- `src/main.rs`
- `tests/pipeline.rs`

### Work

- Construct capture and inference components from validated settings.
- Repeatedly capture a frame and run object detection.
- Keep display code optional and absent from the headless path.
- Stop cleanly on end-of-stream or a shutdown request.
- Propagate component failures without silently ignoring them.
- Add integration tests using deterministic capture and inference test doubles.

### Completion criteria

- Controlled frames can be processed from capture through inference.
- The executable works without a graphical environment.
- Integration tests cover successful processing, end-of-stream, shutdown, and component errors.
- `cargo fmt --check` and `cargo test` pass.

## Step 5: Display and Jetson Validation

### Files

- `src/output.rs`
- `src/output/display.rs`

### Work

- Draw bounding boxes, class labels, and confidence values on processed frames.
- Resize only the displayed representation to the configured dimensions.
- Open a window only when display is enabled.
- Allow the user to request shutdown from the display window.
- Build in release mode on the Jetson Nano.
- Validate camera access, model loading, and CUDA or TensorRT inference on the target device.

### Completion criteria

- Headless execution works without display dependencies at runtime.
- Display mode shows annotated frames at the requested dimensions.
- The application shuts down cleanly from both modes.
- A Jetson Nano smoke test confirms camera capture and accelerated inference.
- `cargo fmt --check`, `cargo test`, and `cargo build --release` pass in their supported environments.

## Working rule

Start only one step at a time. For each behavior, first add a focused failing test, implement the smallest code that makes it pass, and rerun the complete test suite before moving forward.
