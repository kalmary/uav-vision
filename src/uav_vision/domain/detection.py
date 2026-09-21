from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        coordinates = (self.left, self.top, self.right, self.bottom)
        if not all(_is_finite_number(value) for value in coordinates):
            raise ValueError("Bounding box coordinates must be finite numbers")
        if self.left < 0 or self.top < 0 or self.right < 0 or self.bottom < 0:
            raise ValueError("Bounding box coordinates must be non-negative")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("Bounding box must have a positive area")


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise ValueError("Class identifier must be non-negative")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("Class name must be non-empty")
        if not _is_finite_number(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be a finite probability")
        if not isinstance(self.bounding_box, BoundingBox):
            raise ValueError("Detection must have a bounding box")
