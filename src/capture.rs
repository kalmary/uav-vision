// Responsibility: Define the frame-source boundary used by the processing pipeline.
// Input: Capture settings and requests for the next frame.
// Output: Application-owned frames, end-of-stream state, or capture errors.

pub mod camera;

use crate::domain::{Frame, FrameError};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub trait FrameSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureError {
    InvalidSource,
    Open { source: String, message: String },
    Read { message: String },
    Decode { message: String },
    InvalidFrame(FrameError),
    SequenceExhausted,
}

impl Display for CaptureError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSource => write!(formatter, "camera source cannot be empty"),
            Self::Open { source, message } => {
                write!(formatter, "failed to open camera '{source}': {message}")
            }
            Self::Read { message } => write!(formatter, "failed to read camera frame: {message}"),
            Self::Decode { message } => {
                write!(formatter, "failed to decode camera frame: {message}")
            }
            Self::InvalidFrame(error) => write!(formatter, "invalid camera frame: {error}"),
            Self::SequenceExhausted => write!(formatter, "camera frame sequence is exhausted"),
        }
    }
}

impl Error for CaptureError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidFrame(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FrameError> for CaptureError {
    fn from(error: FrameError) -> Self {
        Self::InvalidFrame(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_error_identifies_failed_operation() {
        let error = CaptureError::Open {
            source: "2".to_string(),
            message: "permission denied".to_string(),
        };

        assert_eq!(
            error.to_string(),
            "failed to open camera '2': permission denied"
        );
    }
}
