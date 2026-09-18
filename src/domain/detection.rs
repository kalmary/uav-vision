// Responsibility: Represent an object class, confidence score, and image-space bounding box for one prediction.
// Input: Model output translated from Ultralytics inference results.
// Output: Library-independent detection values used by the pipeline and display.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl BoundingBox {
    pub fn new(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Result<Self, BoundingBoxError> {
        let coordinates = [x_min, y_min, x_max, y_max];
        let are_valid = coordinates
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && x_max > x_min
            && y_max > y_min;

        if !are_valid {
            return Err(BoundingBoxError::InvalidCoordinates);
        }

        Ok(Self {
            x_min,
            y_min,
            x_max,
            y_max,
        })
    }

    pub fn width(self) -> f32 {
        self.x_max - self.x_min
    }

    pub fn height(self) -> f32 {
        self.y_max - self.y_min
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundingBoxError {
    InvalidCoordinates,
}

impl Display for BoundingBoxError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "bounding-box coordinates must be finite, non-negative, and ordered"
        )
    }
}

impl Error for BoundingBoxError {}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bounds: BoundingBox,
}

impl Detection {
    pub fn new(
        class_id: usize,
        class_name: impl Into<String>,
        confidence: f32,
        bounds: BoundingBox,
    ) -> Result<Self, DetectionError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(DetectionError::InvalidConfidence(confidence));
        }

        Ok(Self {
            class_id,
            class_name: class_name.into(),
            confidence,
            bounds,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DetectionError {
    InvalidConfidence(f32),
}

impl Display for DetectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfidence(value) => {
                write!(
                    formatter,
                    "detection confidence must be between 0 and 1, got {value}"
                )
            }
        }
    }
}

impl Error for DetectionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_bounding_box_from_ordered_pixel_coordinates() {
        let bounds = BoundingBox::new(10.0, 20.0, 40.0, 70.0).unwrap();

        assert_eq!(bounds.width(), 30.0);
        assert_eq!(bounds.height(), 50.0);
    }

    #[test]
    fn rejects_invalid_bounding_box_coordinates() {
        assert_eq!(
            BoundingBox::new(10.0, 20.0, 5.0, 70.0),
            Err(BoundingBoxError::InvalidCoordinates)
        );
        assert_eq!(
            BoundingBox::new(f32::NAN, 20.0, 40.0, 70.0),
            Err(BoundingBoxError::InvalidCoordinates)
        );
    }

    #[test]
    fn creates_detection_with_valid_confidence() {
        let bounds = BoundingBox::new(10.0, 20.0, 40.0, 70.0).unwrap();
        let detection = Detection::new(2, "aircraft", 0.85, bounds).unwrap();

        assert_eq!(detection.class_id, 2);
        assert_eq!(detection.class_name, "aircraft");
        assert_eq!(detection.confidence, 0.85);
        assert_eq!(detection.bounds, bounds);
    }

    #[test]
    fn rejects_detection_with_invalid_confidence() {
        let bounds = BoundingBox::new(10.0, 20.0, 40.0, 70.0).unwrap();

        assert_eq!(
            Detection::new(2, "aircraft", 1.1, bounds),
            Err(DetectionError::InvalidConfidence(1.1))
        );
        assert!(matches!(
            Detection::new(2, "aircraft", f32::NAN, bounds),
            Err(DetectionError::InvalidConfidence(value)) if value.is_nan()
        ));
    }
}
