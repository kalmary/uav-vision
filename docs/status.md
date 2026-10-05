# Project Status

The Rust implementation was removed. The project is planned as a modular Python application targeting both NVIDIA Jetson Nano cameras and conventional laptop or USB cameras.

| Workplan step | Status |
| --- | --- |
| 1. Python Foundation, Domain Types, and Configuration | Complete |
| 2. Replaceable Camera Capture | Implemented; laptop camera smoke test passed |
| 3. Replaceable YOLO Inference and Processing Mode | Complete |
| 4. Headless Pipeline, Outputs, and Entry Points | Implemented; laptop headless/display smoke tests passed |
| 5. Jetson Nano Deployment and Validation | Prepared locally; Jetson hardware validation pending |
| 6. Configuration, Processing Modes, and Mandatory Logging | 6a–6i implemented locally; Jetson validation deferred |

## Current decisions

- Python replaces Rust for the application implementation.
- All application code belongs under `src/uav_vision/`; only tests, metadata, and documentation live outside `src/`.
- Camera sources, inference providers, processing modes, and outputs use small replaceable interfaces.
- Laptop and USB cameras receive a dedicated entry point backed by the shared pipeline.
- Jetson acceleration uses an Ultralytics-compatible TensorRT engine validated on the target device.
- Ultralytics, NumPy, and platform-appropriate OpenCV builds are the initial runtime packages; pytest, pytest-cov, Ruff, setuptools, wheel, and uv support development.
- Headless capture and inference remain independent from optional display behavior; mandatory basic/debug logging covers detection, semantic segmentation, and depth statistics.

## Work Log

- Preserve optional Jetson display — confirmed the shared `--display` configuration and documented use of JetPack-provided GUI OpenCV without the desktop extra.
- Summarize implementation status — confirmed that Step 1 is complete and camera, inference, pipeline, display implementation, and Jetson validation remain pending.
- Scope the next implementation step — reviewed all Markdown and code, defined the Step 2 capture contract, and requested approval before coding.
- Approve Step 2 implementation — added replaceable OpenCV and GStreamer capture with explicit errors, cleanup, and automated tests.
- Continue with Step 3 — reviewed the documentation and implemented replaceable YOLO inference and detection processing test-first.
- Audit Step 3 completeness — confirmed all planned code is present through independent review and a real nano-model smoke test.
- Implement Step 4 — added JSON and display outputs, the headless pipeline, application composition, and both console entry points.
- Correct generated file scope — removed redundant root scripts and left `docs/todo_later.md` untouched as requested.
- Report remaining Step 4 work — completed final review, CLI smoke tests, and compatibility verification.
- Assess Step 5 hardware needs — separated locally preparable deployment work from validation that requires a physical Jetson Nano.
- Prepare Step 5 without hardware — documented laptop use and a Jetson deployment and validation runbook with hardware-only results left pending.
- Correct the output contract — removed per-frame JSON output and retained display output while deferring filtered headless logging.
- Audit the output correction — confirmed code, tests, and documentation no longer use per-frame JSON output.
- Replan remaining work — split Jetson preparation from pending hardware validation and defined structured Step 6 configuration, mode, and logging stages.
- Audit Step 6 quality — confirmed broad guideline compliance and identified sequencing and contract gaps to correct before implementation.
- Correct Step 6 sequencing — reordered contracts, logging, filtering, CLI migration, and mode work while defining diagnostics ownership.
- Resolve test type warnings — narrowed optional display settings and locally suppressed intentional invalid-argument cases.
- Complete the Step 6c review — verified exception preservation and traceback credential redaction, then completed the 6a–6c integration checks.
- Refine the remaining workplan — specified grouped camera/device/model/display/FPS CLI contracts, model-specific inference sizing, and logically ordered Steps 6e–6i without changing code.
- Correct startup-reporting requirements — required every entry point to resolve packaged defaults and the selected YOLO model before logging the complete effective configuration.
- Investigate Git synchronization issues — began evidence gathering and requested the exact failing pull or push command and complete error output without running Git operations.
- Diagnose Git synchronization failure — confirmed that `origin/main` exists and local `main` lacks only its upstream tracking configuration.
- Identify divergent Git history — confirmed tracking is fixed while local and remote `main` require history comparison before choosing merge or rebase.
- Compare divergent branches — found four similarly named commits on each side with different hashes and deferred reconciliation until their final tree contents are compared.
- Locate the branch-content difference — confirmed remote `main` differs only by its tracked root-level `yolo26n.pt`, while local model files are untracked under `models/`.
- Define model-weight cleanup — selected a repository-wide `*.pt` ignore rule and a protected update of the outdated remote history, pending design approval.
- Ignore model weights — added a repository-wide `*.pt` rule and verified local YOLO weight files no longer appear as untracked content.
- Revise delegated-agent models — scoped the workflow away from Sol 6 and Sol 6.1 in favor of task-appropriate models.
- Refine delegation policy — selected Sol 5.6 high for high-level work, Terra 6 for well-specified coding, and detailed task descriptions for reliable execution.
- Approve the agent model policy — updated `AGENTS.md` with the agreed model selection and delegation requirements.
- Audit the agent model policy — confirmed every requested model-selection, reasoning-effort, task-detail, and verification rule is present.
- Audit implementation progress — verified the detection pipeline and Steps 6a–6e coverage, then identified unfinished runtime modes, scheduling, integration, and hardware validation.
- Prepare the project test environment — synchronized the declared test group into `.venv` and verified local pytest and Ruff executables.
- Select Step 6f — approved model-specific inference sizing, deterministic FPS limiting, and measured FPS reporting as the next implementation stage.
- Implement Step 6f — added validated model input sizes, explicit Ultralytics sizing, uncapped or limited processing, and FPS logging and display.
- Correct test organization — merged step-named tests into the existing CLI test module and retained component-focused coverage.
- Restore model storage contract — routed every packaged YOLO size through `models/` for both loading and Ultralytics downloads, with component tests and documentation aligned.
- Remove frame sequence state — eliminated sequence counters and values from capture, domain frames, logs, tests, and documentation while retaining timestamps and FPS diagnostics.
- Audit implemented Step 6 work — verified Steps 6a–6f against their contracts and identified remaining logging-redaction, startup-reporting, scheduler-responsiveness, and model-selection verification gaps.
- Close Step 6 audit gaps — fixed credential redaction and startup completeness, made capped waits responsive, unified model identifier and input size, enforced frame-aligned detections, and synchronized deployment documentation.
- Preserve display aspect ratio — made optional display output uniformly fit frames within the configured resolution using black padding while leaving headless frames unchanged.
- Review remaining work — confirmed segmentation, depth, final integration documentation, hardware validation, and Git reconciliation are still outstanding.
- Implement Step 6g — added YOLO26 semantic segmentation models, provider conversion, processing, mode composition, unique-class logging, deterministic display overlays, and real-model verification.
- Implement Step 6h — added source-aligned metric depth inference, immutable maps and cached statistics, CLI and pipeline composition, logging, range-safe Viridis display, and tests with a real nano-model CPU smoke check; Jetson validation remains pending.
- Implement local Step 6i — refreshed runtime documentation, corrected logging choices in CLI help, strengthened entry-point and packaged-default tests, and verified all three modes with the real laptop camera headlessly and with display; Jetson work remains deferred.
- Separate installation from execution — kept dependency profiles and the test group in installation commands and documented plain `uv run` commands, with a Python 3.12 pin for stable local execution.
- Preserve existing entry points — retained both installed console aliases without adding direct-script wrappers or repeating extras in run examples.
- Clarify default configuration ownership — documented the existing packaged application and YOLO files as the source of runtime defaults, with custom files optional and CLI values acting only as explicit overrides.
- Switch configuration to YAML — replaced packaged JSON defaults with `app.yaml` and `yolo.yaml`, recorded PyYAML as a runtime dependency, retained precedence and validation, updated examples, and verified safe parsing plus isolated wheel installation.
- Verify CLI retention — confirmed all CLI options remain available and explicit arguments override YAML defaults while omitted arguments retain configuration values; all CLI tests passed.
