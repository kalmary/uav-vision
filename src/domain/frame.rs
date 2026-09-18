// Responsibility: Represent one captured image together with its dimensions, pixel format, sequence, and capture time.
// Input: Pixel data and metadata produced by a camera source.
// Output: An owned frame that can be consumed by inference and optional display code.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::SystemTime;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    Gray8,
    Rgb8,
    Bgr8,
}

impl PixelFormat {
    fn channel_count(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Rgb8 | Self::Bgr8 => 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub sequence: u64,
    pub captured_at: SystemTime,
}

impl Frame {
    pub fn new(
        data: Vec<u8>,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        sequence: u64,
        captured_at: SystemTime,
    ) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::InvalidDimensions { width, height });
        }

        let expected = width as usize * height as usize * pixel_format.channel_count();
        if data.len() != expected {
            return Err(FrameError::UnexpectedBufferSize {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            data,
            width,
            height,
            pixel_format,
            sequence,
            captured_at,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameError {
    InvalidDimensions { width: u32, height: u32 },
    UnexpectedBufferSize { expected: usize, actual: usize },
}

impl Display for FrameError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(
                    formatter,
                    "frame dimensions must be non-zero, got {width}x{height}"
                )
            }
            Self::UnexpectedBufferSize { expected, actual } => write!(
                formatter,
                "frame buffer has {actual} bytes, expected {expected}"
            ),
        }
    }
}

impl Error for FrameError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn creates_frame_when_buffer_matches_dimensions() {
        let captured_at = UNIX_EPOCH + Duration::from_secs(12);
        let frame = Frame::new(vec![7; 18], 3, 2, PixelFormat::Rgb8, 4, captured_at).unwrap();

        assert_eq!(frame.data, vec![7; 18]);
        assert_eq!(frame.width, 3);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.pixel_format, PixelFormat::Rgb8);
        assert_eq!(frame.sequence, 4);
        assert_eq!(frame.captured_at, captured_at);
    }

    #[test]
    fn rejects_frame_with_wrong_buffer_size() {
        let error = Frame::new(vec![0; 5], 2, 1, PixelFormat::Bgr8, 0, UNIX_EPOCH).unwrap_err();

        assert_eq!(
            error,
            FrameError::UnexpectedBufferSize {
                expected: 6,
                actual: 5,
            }
        );
    }

    #[test]
    fn rejects_frame_with_zero_dimension() {
        let error = Frame::new(Vec::new(), 0, 1, PixelFormat::Gray8, 0, UNIX_EPOCH).unwrap_err();

        assert_eq!(
            error,
            FrameError::InvalidDimensions {
                width: 0,
                height: 1,
            }
        );
    }
}
