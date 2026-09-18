# UAV Vision Architecture

## Goal

Run real-time object detection in a single Rust process on an NVIDIA Jetson Nano. The application captures camera frames, performs Ultralytics YOLO inference, returns class-labelled bounding boxes, and optionally displays annotated frames.

## Data flow

1. `main` receives command-line arguments.
2. `config` validates them and produces runtime settings.
3. `app` builds the camera, detector, and optional display from those settings.
4. `pipeline` repeatedly requests a frame from `capture`.
5. `inference` converts the frame into YOLO input and returns domain detections.
6. `output` optionally overlays detections and displays the processed frame.
7. The pipeline stops on end-of-stream, a user request, or an unrecoverable error.

## Module ownership

- `config`: Command-line parsing and validated runtime settings.
- `domain`: Application-owned frame and detection data shared across module boundaries.
- `capture`: Camera access and conversion into application frames.
- `inference`: Ultralytics YOLO model loading and object detection.
- `output`: Optional annotation and window display.
- `pipeline`: Frame-by-frame processing and shutdown control.
- `app`: Construction of concrete components and application lifecycle.

## Boundaries

- Vendor-specific Ultralytics and ONNX Runtime types stay inside `inference`.
- Camera-library types stay inside `capture`.
- Windowing and drawing-library types stay inside `output`.
- Other modules exchange only types owned by `domain` and `config`.
- Headless operation is represented by the absence of a display component, not by a second processing pipeline.

## Planned implementation order

1. Define domain data and validated configuration.
2. Implement camera capture with deterministic unit tests for configuration and conversion logic.
3. Implement YOLO inference behind the `inference` boundary.
4. Implement optional annotation and display.
5. Connect the components through the pipeline and application entry point.
6. Add integration tests for headless processing and graceful termination.

## Verification

- `cargo fmt --check` validates formatting.
- `cargo test` validates unit and integration behavior.
- A Jetson Nano smoke test validates camera access, model loading, TensorRT or CUDA execution, headless operation, and optional display.
