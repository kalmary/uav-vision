// Responsibility: Expose application-owned data exchanged between capture, inference, pipeline, and output modules.
// Input: Frames from capture and raw prediction values translated by inference.
// Output: Stable frame, bounding-box, and detection representations independent of external libraries.

mod detection;
mod frame;

pub use detection::{BoundingBox, BoundingBoxError, Detection, DetectionError};
pub use frame::{Frame, FrameError, PixelFormat};
