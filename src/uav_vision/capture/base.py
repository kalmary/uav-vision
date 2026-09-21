from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from uav_vision.domain import Frame


class CaptureError(RuntimeError):
    """Base class for camera capture failures."""


class CaptureInitializationError(CaptureError):
    """Raised when a camera source cannot be opened."""


class CaptureReadError(CaptureError):
    """Raised when a camera source cannot provide a frame."""


@runtime_checkable
class FrameSource(Protocol):
    """Read frames until a local file reaches its natural end.

    ``read()`` returns ``None`` only at local-file EOF. Live-source failures and
    reads after ``close()`` raise ``CaptureReadError``. ``close()`` is idempotent.
    """

    def read(self) -> Optional[Frame]: ...

    def close(self) -> None: ...


def frame_from_image(image: np.ndarray, sequence: int) -> Frame:
    try:
        return Frame(
            image=image,
            sequence=sequence,
            captured_at=datetime.now(timezone.utc),
        )
    except ValueError as error:
        raise CaptureReadError("Camera returned an invalid image") from error
