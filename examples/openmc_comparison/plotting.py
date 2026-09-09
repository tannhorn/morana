"""Shared plotting conventions for the OpenMC–Morana comparison case."""

from __future__ import annotations

MATERIAL_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "fuel": (61, 70, 80),
    "nak": (240, 178, 60),
    "ss304": (105, 116, 125),
    "sodium": (142, 202, 230),
    "graphite": (216, 207, 155),
    "zirconium": (27, 94, 129),
}


def material_color_hex(material_name: str) -> str:
    """Return a shared case-material color in Matplotlib's hex notation."""

    red, green, blue = MATERIAL_COLORS_RGB[material_name]
    return f"#{red:02x}{green:02x}{blue:02x}"
