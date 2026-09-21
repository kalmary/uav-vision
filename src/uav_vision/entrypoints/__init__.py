from typing import Optional, Sequence

from uav_vision.config.settings import CaptureSettings


def _run(
    arguments: Optional[Sequence[str]],
    *,
    prog: str,
    capture_defaults: Optional[CaptureSettings] = None,
) -> int:
    from uav_vision import app
    from uav_vision.config import cli

    settings = cli.parse_args(
        arguments,
        prog=prog,
        capture_defaults=capture_defaults,
    )
    try:
        app.run(settings)
    except KeyboardInterrupt:
        return 130
    return 0
