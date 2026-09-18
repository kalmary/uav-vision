// Responsibility: Open the Jetson-connected camera and convert captured images into application frames.
// Input: Camera identifier, requested dimensions, frame rate, and capture requests.
// Output: Ordered frames or a concrete camera initialization or read error.

use crate::capture::{CaptureError, FrameSource};
use crate::config::CaptureConfig;
use crate::domain::{Frame, PixelFormat};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
use nokhwa::{Buffer, Camera};
use std::fmt::{Debug, Formatter};
use std::time::SystemTime;

pub struct CameraSource {
    camera: Camera,
    sequence: u64,
}

impl CameraSource {
    pub fn open(config: &CaptureConfig) -> Result<Self, CaptureError> {
        let index = camera_index(&config.source)?;
        let format = CameraFormat::new_from(
            config.width,
            config.height,
            FrameFormat::MJPEG,
            config.frames_per_second,
        );
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
        let mut camera = Camera::new(index, requested).map_err(|error| CaptureError::Open {
            source: config.source.clone(),
            message: error.to_string(),
        })?;

        camera.open_stream().map_err(|error| CaptureError::Open {
            source: config.source.clone(),
            message: error.to_string(),
        })?;

        Ok(Self {
            camera,
            sequence: 0,
        })
    }
}

impl Debug for CameraSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CameraSource")
            .field("sequence", &self.sequence)
            .finish_non_exhaustive()
    }
}

impl FrameSource for CameraSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
        let next_sequence = self
            .sequence
            .checked_add(1)
            .ok_or(CaptureError::SequenceExhausted)?;
        let buffer = self.camera.frame().map_err(|error| CaptureError::Read {
            message: error.to_string(),
        })?;
        let frame = frame_from_buffer(&buffer, self.sequence, SystemTime::now())?;

        self.sequence = next_sequence;
        Ok(Some(frame))
    }
}

fn camera_index(source: &str) -> Result<CameraIndex, CaptureError> {
    let source = source.trim();
    if source.is_empty() {
        return Err(CaptureError::InvalidSource);
    }

    Ok(match source.parse() {
        Ok(index) => CameraIndex::Index(index),
        Err(_) => CameraIndex::String(source.to_string()),
    })
}

fn frame_from_buffer(
    buffer: &Buffer,
    sequence: u64,
    captured_at: SystemTime,
) -> Result<Frame, CaptureError> {
    let image = buffer
        .decode_image::<RgbFormat>()
        .map_err(|error| CaptureError::Decode {
            message: error.to_string(),
        })?;
    let width = image.width();
    let height = image.height();

    Frame::new(
        image.into_raw(),
        width,
        height,
        PixelFormat::Rgb8,
        sequence,
        captured_at,
    )
    .map_err(CaptureError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::CaptureError;
    use crate::config::CaptureConfig;
    use crate::domain::PixelFormat;
    use nokhwa::Buffer;
    use nokhwa::utils::{CameraIndex, FrameFormat, Resolution};
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn parses_numeric_camera_source_as_device_index() {
        assert_eq!(camera_index("2").unwrap(), CameraIndex::Index(2));
    }

    #[test]
    fn preserves_named_camera_source() {
        assert_eq!(
            camera_index("/dev/video0").unwrap(),
            CameraIndex::String("/dev/video0".to_string())
        );
    }

    #[test]
    fn rejects_empty_camera_source_before_opening_device() {
        let config = CaptureConfig {
            source: String::new(),
            width: 640,
            height: 480,
            frames_per_second: 30,
        };

        let error = CameraSource::open(&config).unwrap_err();

        assert_eq!(error, CaptureError::InvalidSource);
    }

    #[test]
    fn converts_raw_rgb_buffer_into_application_frame() {
        let buffer = Buffer::new(
            Resolution::new(2, 1),
            &[255, 0, 0, 0, 255, 0],
            FrameFormat::RAWRGB,
        );
        let captured_at = UNIX_EPOCH + Duration::from_secs(3);

        let frame = frame_from_buffer(&buffer, 7, captured_at).unwrap();

        assert_eq!(frame.data, vec![255, 0, 0, 0, 255, 0]);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 1);
        assert_eq!(frame.pixel_format, PixelFormat::Rgb8);
        assert_eq!(frame.sequence, 7);
        assert_eq!(frame.captured_at, captured_at);
    }

    #[test]
    fn reports_invalid_camera_buffer() {
        let buffer = Buffer::new(Resolution::new(2, 1), &[255, 0, 0], FrameFormat::RAWRGB);

        let error = frame_from_buffer(&buffer, 0, UNIX_EPOCH).unwrap_err();

        assert!(matches!(error, CaptureError::Decode { .. }));
    }
}
