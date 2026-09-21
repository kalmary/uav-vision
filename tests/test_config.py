from pathlib import Path

import pytest

from uav_vision.config.models import ModelSize, model_path_for
from uav_vision.config.settings import (
    AppSettings,
    CameraType,
    CaptureSettings,
    DisplaySettings,
    InferenceSettings,
    ProcessingSettings,
    ProcessingType,
)


def test_default_settings_select_headless_detection_with_the_nano_model():
    settings = AppSettings()

    assert settings.capture.camera_type is CameraType.OPENCV
    assert settings.capture.source == 0
    assert settings.processing.processing_type is ProcessingType.DETECTION
    assert settings.inference.model_size is ModelSize.NANO
    assert settings.display is None


def test_model_path_for_the_nano_model_is_centralized():
    assert model_path_for(ModelSize.NANO) == "yolo26n.pt"


@pytest.mark.parametrize(
    ("size", "path"),
    [
        (ModelSize.SMALL, "yolo26s.pt"),
        (ModelSize.MEDIUM, "yolo26m.pt"),
        (ModelSize.LARGE, "yolo26l.pt"),
        (ModelSize.XLARGE, "yolo26x.pt"),
    ],
)
def test_model_path_for_maps_each_supported_model_size(size, path):
    assert model_path_for(size) == path


def test_explicit_model_path_replaces_the_model_size():
    inference = InferenceSettings(model_size=None, model_path=Path("model.engine"))

    assert inference.model_size is None
    assert inference.model_path == Path("model.engine")


def test_inference_rejects_selecting_a_model_size_and_path_together():
    with pytest.raises(ValueError, match="either"):
        InferenceSettings(model_size=ModelSize.NANO, model_path=Path("model.engine"))


@pytest.mark.parametrize("model_path", [Path(), Path("   ")])
def test_inference_rejects_an_empty_model_path(model_path):
    with pytest.raises(ValueError, match="path"):
        InferenceSettings(model_size=None, model_path=model_path)


@pytest.mark.parametrize(
    "dimensions",
    [(0, 720), (1280, 0), (-1, 720), (640.5, 480), (True, 480)],
)
def test_display_rejects_non_positive_dimensions(dimensions):
    with pytest.raises(ValueError, match="positive"):
        DisplaySettings(*dimensions)


@pytest.mark.parametrize("source", ["", "   "])
def test_capture_rejects_an_empty_camera_source(source):
    with pytest.raises(ValueError, match="source"):
        CaptureSettings(source=source)


def test_capture_rejects_a_negative_camera_index():
    with pytest.raises(ValueError, match="source"):
        CaptureSettings(source=-1)


def test_gstreamer_capture_requires_a_pipeline_string():
    with pytest.raises(ValueError, match="GStreamer"):
        CaptureSettings(camera_type=CameraType.GSTREAMER, source=0)


@pytest.mark.parametrize("device", ["   ", 0])
def test_inference_rejects_an_invalid_device(device):
    with pytest.raises(ValueError, match="device"):
        InferenceSettings(device=device)


@pytest.mark.parametrize(
    "settings",
    [
        lambda: CaptureSettings(camera_type="gstreamer", source="pipeline"),
        lambda: ProcessingSettings(processing_type="unsupported"),
        lambda: InferenceSettings(model_size="nano"),
    ],
)
def test_settings_reject_non_enum_selections(settings):
    with pytest.raises(ValueError, match="selection"):
        settings()


def test_inference_rejects_a_model_path_that_is_not_a_path():
    with pytest.raises(ValueError, match="path"):
        InferenceSettings(model_size=None, model_path="model.engine")


def test_settings_keep_capture_processing_inference_and_display_separate():
    settings = AppSettings(
        capture=CaptureSettings(camera_type=CameraType.GSTREAMER, source="pipeline"),
        processing=ProcessingSettings(),
        inference=InferenceSettings(),
        display=DisplaySettings(1280, 720),
    )

    assert settings.capture.source == "pipeline"
    assert settings.display.width == 1280
