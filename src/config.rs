// Responsibility: Own validated runtime settings shared by the application components.
// Input: Raw values produced by the command-line parser.
// Output: Camera, processing, model, device, and display settings in valid combinations.

pub mod cli;

use std::path::PathBuf;
use std::str::FromStr;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppConfig {
    pub capture: CaptureConfig,
    pub processing: ProcessingType,
    pub inference: InferenceConfig,
    pub display: Option<DisplayConfig>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaptureConfig {
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub frames_per_second: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessingType {
    Detection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InferenceConfig {
    pub model: ModelSelection,
    pub device: InferenceDevice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelSelection {
    Size(ModelSize),
    Path(PathBuf),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelSize {
    Nano,
    Small,
    Medium,
    Large,
    ExtraLarge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceDevice {
    Cpu,
    Cuda(u32),
    TensorRt(u32),
}

impl FromStr for InferenceDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "cpu" {
            return Ok(Self::Cpu);
        }

        if let Some(index) = value.strip_prefix("cuda:") {
            return parse_device_index(index, "cuda:INDEX").map(Self::Cuda);
        }

        if let Some(index) = value.strip_prefix("tensorrt:") {
            return parse_device_index(index, "tensorrt:INDEX").map(Self::TensorRt);
        }

        Err(device_error(value))
    }
}

fn parse_device_index(value: &str, expected: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("invalid inference device index; expected {expected}"))
}

fn device_error(value: &str) -> String {
    format!("invalid inference device '{value}'; expected cpu, cuda:INDEX, or tensorrt:INDEX")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DisplayConfig {
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn parses_supported_inference_devices() {
        assert_eq!(InferenceDevice::from_str("cpu"), Ok(InferenceDevice::Cpu));
        assert_eq!(
            InferenceDevice::from_str("cuda:2"),
            Ok(InferenceDevice::Cuda(2))
        );
        assert_eq!(
            InferenceDevice::from_str("tensorrt:1"),
            Ok(InferenceDevice::TensorRt(1))
        );
    }

    #[test]
    fn rejects_invalid_inference_device() {
        let error = InferenceDevice::from_str("cuda:x").unwrap_err();

        assert!(error.contains("cuda:INDEX"));
    }
}
