from typing import Tuple

_BASIC_COLORS = (
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("red", (255, 0, 0)),
    ("orange", (255, 165, 0)),
    ("yellow", (255, 255, 0)),
    ("green", (0, 128, 0)),
    ("blue", (0, 0, 255)),
    ("purple", (128, 0, 128)),
    ("pink", (255, 192, 203)),
    ("brown", (165, 42, 42)),
)


def nearest_color_name(rgb: Tuple[int, int, int]) -> str:
    return min(
        _BASIC_COLORS,
        key=lambda color: sum(
            (channel - reference) ** 2 for channel, reference in zip(rgb, color[1])
        ),
    )[0]


def segmentation_color(class_id: int) -> Tuple[int, int, int]:
    return (
        (37 * class_id + 53) % 256,
        (17 * class_id + 97) % 256,
        (29 * class_id + 193) % 256,
    )


def test_nearest_color_name_matches_basic_colors():
    cases = (
        ((0, 0, 0), "black"),
        ((255, 255, 255), "white"),
        ((255, 0, 0), "red"),
        ((255, 165, 0), "orange"),
        ((255, 255, 0), "yellow"),
        ((0, 128, 0), "green"),
        ((0, 0, 255), "blue"),
        ((128, 0, 128), "purple"),
        ((255, 192, 203), "pink"),
        ((165, 42, 42), "brown"),
    )
    for color, expected in cases:
        assert nearest_color_name(color) == expected


def test_nearest_color_name_matches_intermediate_colors():
    assert nearest_color_name((251, 131, 127)) == "pink"
    assert nearest_color_name((140, 216, 56)) == "yellow"
    assert nearest_color_name((193, 97, 53)) == "brown"
    assert nearest_color_name((58, 62, 22)) == "black"


def test_nearest_color_name_breaks_distance_ties_by_palette_order():
    assert nearest_color_name((0, 64, 0)) == "black"
