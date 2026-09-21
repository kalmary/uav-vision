# Project Status

The Rust implementation was removed. The project is planned as a modular Python application targeting both NVIDIA Jetson Nano cameras and conventional laptop or USB cameras.

| Workplan step | Status |
| --- | --- |
| 1. Python Foundation, Domain Types, and Configuration | Complete |
| 2. Replaceable Camera Capture | Implemented; hardware smoke test pending |
| 3. Replaceable YOLO Inference and Processing Mode | Not started |
| 4. Headless Pipeline, Outputs, and Entry Points | Not started |
| 5. Jetson Nano Deployment and Validation | Not started |

## Current decisions

- Python replaces Rust for the application implementation.
- All application code belongs under `src/uav_vision/`; only tests, metadata, and documentation live outside `src/`.
- Camera sources, inference providers, processing modes, and outputs use small replaceable interfaces.
- Laptop and USB cameras receive a dedicated entry point backed by the shared pipeline.
- Jetson acceleration uses an Ultralytics-compatible TensorRT engine validated on the target device.
- Ultralytics, NumPy, and platform-appropriate OpenCV builds are the initial runtime packages; pytest, pytest-cov, Ruff, setuptools, wheel, and uv support development.
- Headless JSON Lines output remains independent from optional display behavior.

## Work Log

- Define the project task — documented the Jetson Nano UAV camera, YOLO detection, optional display, and CLI requirements in `AGENTS.md`.
- Test the working setup — checked the repository workflow without adding a project change.
- Expand agent guidance — added model-tier selection and agent-workflow rules to `AGENTS.md`.
- Plan the architecture — created the Rust project structure with responsibility, input, and output descriptions for scaffold files.
- Recheck repository instructions — aligned the planned architecture with the repository module and engineering rules.
- Build the planned scaffold — added the requested folders and initial Rust files.
- Keep the solution Rust-only — selected a Rust-native capture and inference direction in the workplan.
- Divide implementation into steps — structured the project into five independently testable stages.
- Save the implementation plan — added `docs/workplan.md`.
- Implement Step 1 — added domain types, CLI configuration, validation, and unit tests.
- Investigate incomplete compilation — identified scaffold placeholders that had not yet been implemented.
- Implement Step 2 — added camera capture, frame conversion, capture errors, Linux V4L2 configuration, and unit tests.
- Track implementation status — added this status document with the state of every workplan step.
- Add per-prompt change tracking — added this rolling work log and its 50-entry maintenance rule to `AGENTS.md`.
- Add JSON-backed defaults — designed configuration fallback for omitted command-line arguments.
- Consider runtime-loaded defaults — evaluated disk loading, then superseded it before implementation.
- Select compile-time embedding — placed the editable defaults in `src/config/config.json` and embedded them in the binary.
- Approve the configuration design — implemented JSON fallback, CLI precedence, validation, executable parsing, and focused tests.
- Review repository documentation — read all four Markdown files and recorded the current requirements, architecture, status, and work plan.
- Evaluate Rust GPU inference — confirmed Jetson Nano CUDA is viable through its JetPack TensorRT stack, while the current Ultralytics Rust crate cannot provide compatible AArch64 GPU binaries.
- Reset the implementation language — removed the Rust source, tests, Cargo metadata, and generated build artifacts, then replanned the application as modular Python with separate Jetson and USB-camera adapters.
- Refresh ignored files — replaced the obsolete Rust target rule with macOS, Zed, Python, virtual-environment, build, test-cache, coverage, and local-environment patterns.
- Fix the Python source boundary — made `src/uav_vision/` the required location for every application module and entry point.
- Select initial Python packages — documented maintained dependencies, mutually exclusive OpenCV variants, and preservation of JetPack-provided GPU libraries in the work plan.
- Implement Python Step 1 — added the package foundation, validated domain and configuration types, CLI parsing, mutually exclusive runtime profiles, dependency groups, and passing tests and Ruff checks.
- Clarify runtime dependencies — documented why each installation must select exactly one desktop, generic-headless, or Jetson platform profile.
- Clarify Python support — kept Python 3.8 as the JetPack 4 compatibility floor while confirming Python 3.12 as the desktop development runtime.
- Explain deployment profiles — documented the planned laptop installation and dependency-preserving JetPack 4 container workflow.
- Preserve optional Jetson display — confirmed the shared `--display` configuration and documented use of JetPack-provided GUI OpenCV without the desktop extra.
- Summarize implementation status — confirmed that Step 1 is complete and camera, inference, pipeline, display implementation, and Jetson validation remain pending.
- Scope the next implementation step — reviewed all Markdown and code, defined the Step 2 capture contract, and requested approval before coding.
- Approve Step 2 implementation — added replaceable OpenCV and GStreamer capture with explicit errors, cleanup, and automated tests.
