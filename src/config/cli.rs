// Responsibility: Define and parse the command-line interface without constructing runtime components.
// Input: Command-line argument strings and embedded JSON defaults.
// Output: Raw configuration values or a descriptive command-line validation error.

use crate::config::{
    AppConfig, CaptureConfig, DisplayConfig, InferenceConfig, InferenceDevice, ModelSelection,
    ModelSize, ProcessingType,
};
use clap::error::ErrorKind;
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::ffi::OsString;
use std::path::PathBuf;
use std::str::FromStr;

const DEFAULT_ARGUMENTS: &str = include_str!("config.json");

#[derive(Debug, Parser)]
#[command(
    name = "uav-vision-rs",
    version,
    about = "Real-time UAV object detection"
)]
struct Arguments {
    #[arg(long)]
    camera_source: Option<String>,

    #[arg(
        long,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    camera_width: Option<u32>,

    #[arg(
        long,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    camera_height: Option<u32>,

    #[arg(
        long,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    camera_fps: Option<u32>,

    #[arg(long, value_enum)]
    processing: Option<ProcessingArgument>,

    #[arg(long, conflicts_with = "model_size")]
    model: Option<PathBuf>,

    #[arg(long, value_enum)]
    model_size: Option<ModelSizeArgument>,

    #[arg(long)]
    device: Option<InferenceDevice>,

    #[arg(long)]
    display: bool,

    #[arg(
        long,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    display_width: Option<u32>,

    #[arg(
        long,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    display_height: Option<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum ProcessingArgument {
    Detection,
}

#[derive(Clone, Copy, Debug, Deserialize, ValueEnum)]
enum ModelSizeArgument {
    #[serde(rename = "n", alias = "nano")]
    #[value(name = "n", alias = "nano")]
    Nano,
    #[serde(rename = "s", alias = "small")]
    #[value(name = "s", alias = "small")]
    Small,
    #[serde(rename = "m", alias = "medium")]
    #[value(name = "m", alias = "medium")]
    Medium,
    #[serde(rename = "l", alias = "large")]
    #[value(name = "l", alias = "large")]
    Large,
    #[serde(rename = "x", alias = "extra-large")]
    #[value(name = "x", alias = "extra-large")]
    ExtraLarge,
}

#[derive(Debug, Deserialize)]
struct DefaultArguments {
    camera_source: String,
    camera_width: u32,
    camera_height: u32,
    camera_fps: u32,
    processing: ProcessingArgument,
    model_size: ModelSizeArgument,
    device: String,
    display: bool,
    display_width: u32,
    display_height: u32,
}

pub fn parse() -> Result<AppConfig, clap::Error> {
    parse_from_with_defaults(std::env::args_os(), DEFAULT_ARGUMENTS)
}

pub fn parse_from<I, T>(arguments: I) -> Result<AppConfig, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    parse_from_with_defaults(arguments, DEFAULT_ARGUMENTS)
}

fn parse_from_with_defaults<I, T>(arguments: I, defaults: &str) -> Result<AppConfig, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let defaults: DefaultArguments =
        serde_json::from_str(defaults).map_err(|error| defaults_error(error.to_string()))?;
    validate_defaults(&defaults)?;
    let arguments = Arguments::try_parse_from(arguments)?;

    resolve(arguments, defaults)
}

fn resolve(arguments: Arguments, defaults: DefaultArguments) -> Result<AppConfig, clap::Error> {
    let device = match arguments.device {
        Some(device) => device,
        None => InferenceDevice::from_str(&defaults.device).map_err(defaults_error)?,
    };

    let model = match (arguments.model, arguments.model_size) {
        (Some(path), None) => ModelSelection::Path(path),
        (None, Some(size)) => ModelSelection::Size(size.into()),
        (None, None) => ModelSelection::Size(defaults.model_size.into()),
        (Some(_), Some(_)) => unreachable!("clap rejects conflicting model options"),
    };

    let display_enabled = arguments.display || defaults.display;
    let display = display_enabled.then_some(DisplayConfig {
        width: arguments.display_width.unwrap_or(defaults.display_width),
        height: arguments.display_height.unwrap_or(defaults.display_height),
    });

    Ok(AppConfig {
        capture: CaptureConfig {
            source: arguments.camera_source.unwrap_or(defaults.camera_source),
            width: arguments.camera_width.unwrap_or(defaults.camera_width),
            height: arguments.camera_height.unwrap_or(defaults.camera_height),
            frames_per_second: arguments.camera_fps.unwrap_or(defaults.camera_fps),
        },
        processing: arguments.processing.unwrap_or(defaults.processing).into(),
        inference: InferenceConfig { model, device },
        display,
    })
}

fn validate_defaults(defaults: &DefaultArguments) -> Result<(), clap::Error> {
    if defaults.camera_width == 0
        || defaults.camera_height == 0
        || defaults.camera_fps == 0
        || defaults.display_width == 0
        || defaults.display_height == 0
    {
        return Err(defaults_error("dimensions and frame rate must be positive"));
    }

    Ok(())
}

fn defaults_error(message: impl Into<String>) -> clap::Error {
    clap::Error::raw(
        ErrorKind::ValueValidation,
        format!("invalid embedded configuration: {}", message.into()),
    )
}

impl From<ProcessingArgument> for ProcessingType {
    fn from(value: ProcessingArgument) -> Self {
        match value {
            ProcessingArgument::Detection => Self::Detection,
        }
    }
}

impl From<ModelSizeArgument> for ModelSize {
    fn from(value: ModelSizeArgument) -> Self {
        match value {
            ModelSizeArgument::Nano => Self::Nano,
            ModelSizeArgument::Small => Self::Small,
            ModelSizeArgument::Medium => Self::Medium,
            ModelSizeArgument::Large => Self::Large,
            ModelSizeArgument::ExtraLarge => Self::ExtraLarge,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{InferenceDevice, ModelSelection, ModelSize, ProcessingType};
    use clap::error::ErrorKind;
    use std::path::PathBuf;

    const TEST_DEFAULTS: &str = r#"
    {
        "camera_source": "2",
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 15,
        "processing": "detection",
        "model_size": "s",
        "device": "cuda:0",
        "display": true,
        "display_width": 800,
        "display_height": 600
    }
    "#;

    #[test]
    fn uses_json_defaults_for_omitted_cli_arguments() {
        let config = parse_from_with_defaults(["uav-vision-rs"], TEST_DEFAULTS).unwrap();

        assert_eq!(config.capture.source, "2");
        assert_eq!(config.capture.width, 640);
        assert_eq!(config.capture.height, 480);
        assert_eq!(config.capture.frames_per_second, 15);
        assert_eq!(config.processing, ProcessingType::Detection);
        assert_eq!(
            config.inference.model,
            ModelSelection::Size(ModelSize::Small)
        );
        assert_eq!(config.inference.device, InferenceDevice::Cuda(0));
        assert_eq!(
            config.display,
            Some(DisplayConfig {
                width: 800,
                height: 600,
            })
        );
    }

    #[test]
    fn explicit_cli_arguments_override_json_defaults() {
        let config = parse_from_with_defaults(
            [
                "uav-vision-rs",
                "--camera-source",
                "/dev/video4",
                "--camera-width",
                "1920",
                "--model",
                "models/custom.onnx",
                "--device",
                "tensorrt:1",
            ],
            TEST_DEFAULTS,
        )
        .unwrap();

        assert_eq!(config.capture.source, "/dev/video4");
        assert_eq!(config.capture.width, 1920);
        assert_eq!(config.capture.height, 480);
        assert_eq!(
            config.inference.model,
            ModelSelection::Path(PathBuf::from("models/custom.onnx"))
        );
        assert_eq!(config.inference.device, InferenceDevice::TensorRt(1));
    }

    #[test]
    fn rejects_non_positive_json_defaults() {
        let defaults = TEST_DEFAULTS.replace("\"camera_width\": 640", "\"camera_width\": 0");

        let error = parse_from_with_defaults(["uav-vision-rs"], &defaults).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn uses_headless_nano_defaults() {
        let config = parse_from(["uav-vision-rs"]).unwrap();

        assert_eq!(config.capture.source, "0");
        assert_eq!(config.capture.width, 1280);
        assert_eq!(config.capture.height, 720);
        assert_eq!(config.capture.frames_per_second, 30);
        assert_eq!(config.processing, ProcessingType::Detection);
        assert_eq!(
            config.inference.model,
            ModelSelection::Size(ModelSize::Nano)
        );
        assert_eq!(config.inference.device, InferenceDevice::Cpu);
        assert_eq!(config.display, None);
    }

    #[test]
    fn parses_camera_model_device_and_display_options() {
        let config = parse_from([
            "uav-vision-rs",
            "--camera-source",
            "rtsp://camera/stream",
            "--camera-width",
            "640",
            "--camera-height",
            "480",
            "--camera-fps",
            "20",
            "--processing",
            "detection",
            "--model-size",
            "s",
            "--device",
            "cuda:1",
            "--display",
            "--display-width",
            "800",
            "--display-height",
            "600",
        ])
        .unwrap();

        assert_eq!(config.capture.source, "rtsp://camera/stream");
        assert_eq!(config.capture.width, 640);
        assert_eq!(config.capture.height, 480);
        assert_eq!(config.capture.frames_per_second, 20);
        assert_eq!(config.processing, ProcessingType::Detection);
        assert_eq!(
            config.inference.model,
            ModelSelection::Size(ModelSize::Small)
        );
        assert_eq!(config.inference.device, InferenceDevice::Cuda(1));
        assert_eq!(
            config.display,
            Some(crate::config::DisplayConfig {
                width: 800,
                height: 600,
            })
        );
    }

    #[test]
    fn uses_explicit_model_path() {
        let config = parse_from(["uav-vision-rs", "--model", "models/aircraft.onnx"]).unwrap();

        assert_eq!(
            config.inference.model,
            ModelSelection::Path(PathBuf::from("models/aircraft.onnx"))
        );
    }

    #[test]
    fn rejects_model_path_with_model_size() {
        let error = parse_from([
            "uav-vision-rs",
            "--model",
            "models/aircraft.onnx",
            "--model-size",
            "n",
        ])
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
    }

    #[test]
    fn rejects_zero_camera_dimension() {
        let error = parse_from(["uav-vision-rs", "--camera-width", "0"]).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn rejects_invalid_device() {
        let error = parse_from(["uav-vision-rs", "--device", "cuda:x"]).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }
}
