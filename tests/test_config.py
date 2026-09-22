import pytest

from uav_vision.config.models import ModelSize
from uav_vision.config.settings import (
    AppSettings,
    CameraType,
    CaptureSettings,
    DetectionFilterSettings,
    DisplaySettings,
    InferenceSettings,
    ProcessingSettings,
)


def test_default_settings_keep_display_dimensions_when_display_is_disabled():
    settings = AppSettings()

    assert settings.capture == CaptureSettings(CameraType.OPENCV, 0)
    assert settings.inference == InferenceSettings(ModelSize.NANO, "cpu")
    assert settings.fps == 30
    assert settings.display == DisplaySettings(1280, 720, False)


def test_settings_reject_missing_display_configuration():
    with pytest.raises(ValueError, match="display settings"):
        AppSettings(display=None)


@pytest.mark.parametrize(
    "dimensions",
    [
        (0, 720, False),
        (-1, 720, False),
        (640.5, 720, False),
        (1280, 0, False),
        (1280, -1, False),
        (1280, 480.5, False),
        (640, 480, 1),
    ],
)
def test_display_rejects_invalid_enabled_state_or_dimensions(dimensions):
    with pytest.raises(ValueError):
        DisplaySettings(*dimensions)


@pytest.mark.parametrize("source", ["", "   ", -1])
def test_capture_rejects_invalid_sources(source):
    with pytest.raises(ValueError, match="source"):
        CaptureSettings(source=source)


def test_gstreamer_capture_requires_a_pipeline_string():
    with pytest.raises(ValueError, match="GStreamer"):
        CaptureSettings(CameraType.GSTREAMER, 0)


@pytest.mark.parametrize("device", ["cuda:0", "", None, 0])
def test_inference_accepts_only_cpu_or_cuda(device):
    with pytest.raises(ValueError, match="device"):
        InferenceSettings(device=device)


@pytest.mark.parametrize("fps", [0, -1, 1.5, True])
def test_settings_reject_invalid_fps(fps):
    with pytest.raises(ValueError, match="fps"):
        AppSettings(fps=fps)


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


@pytest.mark.parametrize(
    "settings",
    [
        lambda: DetectionFilterSettings(selected_classes=(True,)),
        lambda: DetectionFilterSettings(selected_classes=(-1,)),
        lambda: DetectionFilterSettings(minimum_confidence=-0.1),
        lambda: DetectionFilterSettings(minimum_confidence=1.1),
        lambda: DetectionFilterSettings(minimum_confidence=True),
        lambda: DetectionFilterSettings(minimum_confidence=float("nan")),
        lambda: DetectionFilterSettings(top_k=0),
        lambda: DetectionFilterSettings(top_k=True),
    ],
)
def test_detection_filter_settings_reject_invalid_values(settings):
    with pytest.raises(ValueError):
        settings()
