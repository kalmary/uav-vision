from typing import Optional, Sequence


def _run(
    arguments: Optional[Sequence[str]],
    *,
    prog: str,
) -> int:
    from uav_vision import app
    from uav_vision.config import cli

    settings = cli.parse_args(arguments, prog=prog)
    try:
        app.run(settings)
    except KeyboardInterrupt:
        return 130
    return 0

