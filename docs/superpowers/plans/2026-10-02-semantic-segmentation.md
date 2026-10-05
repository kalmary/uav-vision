# Semantic Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end semantic segmentation as the second supported processing mode while preserving the existing detection and headless contracts.

**Architecture:** Resolve a mode-specific YOLO26 semantic model, convert the provider's dense class map into the existing application-owned `SegmentationResult`, and process it through the shared pipeline. Logging and display branch on the result variant; display performs visualization only and never changes the class map or headless frame.

**Tech Stack:** Python 3.8+, NumPy, Ultralytics, OpenCV display boundary, pytest, Ruff.

**Spec:** `docs/workplan.md`, Step 6g.

## Global Constraints

- Keep all model identifiers under `models/`; default to `models/yolo26n-sem.pt` with input size `640`.
- Preserve detection behavior and public configuration contracts.
- Keep Ultralytics result objects inside the inference adapter.
- Do not import or construct display resources in headless mode.
- Do not mutate source frames or application-owned segmentation results.
- Add component tests to existing test modules; do not create step-named tests.
- Remain compatible with Python 3.8 and add no dependency.
- Do not run Git commands, commit, stage, or push.

## Review Focus

- Missing or multiple provider results must fail with `InferenceResultError`, not leak provider exceptions.
- A missing, non-integer, wrong-dimensional, or source-size-mismatched class map must fail at the adapter boundary.
- Class IDs absent from provider metadata must fail rather than receive fabricated names.
- Display overlays must remain deterministic and must not mutate the source frame or class map.
- Detection must retain its existing model selection, filtering, annotation, logging, and headless behavior.

---

### Task 1: Segmentation model resolution and provider adapter

**Files:**
- Modify: `src/uav_vision/config/defaults/yolo.json`
- Modify: `src/uav_vision/inference/ultralytics.py`
- Modify: `src/uav_vision/inference/__init__.py`
- Modify: `tests/test_config_loader.py`
- Modify: `tests/test_inference.py`

**Interfaces:**
- Consumes: `Segmenter.segment(frame: Frame) -> SegmentationResult`, `ModelSettings(identifier, input_size)`, and provider fields `result.semantic_mask.data` plus `result.names`.
- Produces: `UltralyticsSegmenter(settings, model_identifier, input_size, model_factory=None)` implementing `Segmenter`.

- [ ] Add failing configuration tests proving all five segmentation sizes resolve to `models/yolo26{size}-sem.pt` and input size `640`.
- [ ] Add failing adapter tests for model construction with task `semantic`, explicit `imgsz` and device, one and multiple present classes, source-size alignment, provider initialization/run failures, missing or multiple results, missing mask, malformed map, and unknown class IDs.
- [ ] Run the focused tests and confirm they fail because segmentation defaults and `UltralyticsSegmenter` do not exist.
- [ ] Add the five semantic model entries and implement the adapter with application-owned copied data and present-class metadata ordered by class ID.
- [ ] Run `tests/test_config_loader.py` and `tests/test_inference.py` and confirm they pass.

**Ruling:** A valid non-empty dense semantic map always contains at least one class ID. The Step 6g “zero classes” case is represented by malformed/missing provider output and must fail explicitly; class ID `0` is not assumed to mean background because valid datasets may assign it to a foreground category.

### Task 2: Segmentation processor and application/CLI composition

**Files:**
- Create: `src/uav_vision/processing/segmentation.py`
- Modify: `src/uav_vision/processing/__init__.py`
- Modify: `src/uav_vision/app.py`
- Modify: `src/uav_vision/config/cli.py`
- Modify: `tests/test_processing.py`
- Modify: `tests/test_app.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `UltralyticsSegmenter` and `SegmentationResult` from Task 1.
- Produces: `SegmentationProcessor(segmenter, clock=perf_counter)` returning a `ProcessedFrame` with `ProcessingType.SEGMENTATION` and present-class counts in diagnostics.

- [ ] Add failing processor tests for structural protocol compliance, same-frame/result preservation, class counts, timing ownership, and unchanged provider errors.
- [ ] Add failing CLI tests that accept segmentation, resolve its selected model, exclude detection filters, and continue rejecting depth as unavailable.
- [ ] Add failing application tests proving segmentation constructs the semantic adapter and processor for headless and display runs while detection composition remains unchanged.
- [ ] Run the focused tests and confirm the missing processor/composition failures.
- [ ] Implement the processor, export it lazily, enable segmentation in CLI validation, and branch application composition by `ProcessingType`.
- [ ] Run `tests/test_processing.py`, `tests/test_cli.py`, and `tests/test_app.py` and confirm they pass.

### Task 3: Mode-specific logging and deterministic display overlay

**Files:**
- Modify: `src/uav_vision/output/log.py`
- Modify: `src/uav_vision/output/display.py`
- Modify: `tests/test_output.py`
- Modify: `docs/architecture.md`
- Modify: `docs/status.md`

**Interfaces:**
- Consumes: segmentation `ProcessedFrame` values produced by Task 2.
- Produces: `basic` record `frame fps=<value> segmentation.classes=<count>` and a deterministic class-colour overlay with labels for classes present in the map.

- [ ] Add failing logger tests for one and multiple present classes, measured/unavailable FPS, and debug diagnostics.
- [ ] Add failing display tests for deterministic colours and labels, unchanged source/class-map data, configured aspect-ratio padding, and quit handling.
- [ ] Run focused output tests and confirm they fail because outputs handle detection only.
- [ ] Implement result-type branches, integer-safe NumPy blending, deterministic BGR colours, and present-class labels before the existing fit-with-padding display step.
- [ ] Update architecture/status documentation and keep the Work Log at no more than 50 bullets.
- [ ] Run `tests/test_output.py` and confirm it passes.

### Task 4: Integration verification

**Files:**
- Modify only files required by verified integration defects.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a tested semantic path without detection regressions.

- [ ] Run `.venv/bin/python -m pytest -q` with `UV_CACHE_DIR=/private/tmp/uav-vision-uv-cache`.
- [ ] Run `.venv/bin/ruff check src tests`.
- [ ] Run `.venv/bin/ruff format --check src tests`.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_cli.py tests/test_app.py tests/test_inference.py tests/test_processing.py tests/test_output.py` after any fix.
- [ ] Perform a fresh review against Step 6g, the Global Constraints, and Review Focus before reporting completion.

