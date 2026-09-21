from typing import Optional, Sequence

from . import _run


def main(arguments: Optional[Sequence[str]] = None) -> int:
    return _run(arguments, prog="uav-vision")
