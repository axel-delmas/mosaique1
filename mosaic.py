"""Data model for a mosaic: square tiles placed on a rectangular board."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Tile:
    """A square tile. Its size is not stored here: every tile in a Mosaic
    has the same width, given by Mosaic.tile_width.

    color       : (r, g, b), each an int in 0..255
    position    : (x, y) of the tile's CENTER, in mosaic coordinates
    orientation : rotation in degrees, counter-clockwise. Any value is accepted and
                  wrapped into [0, 90): a square looks the same after a 90° turn,
                  so e.g. 91° is stored as 1° and -5° as 85°.
    """
    color: tuple[int, int, int]
    position: tuple[float, float]
    orientation: float = 0.0

    def __post_init__(self):
        if len(self.color) != 3 or not all(0 <= c <= 255 for c in self.color):
            raise ValueError(f"color must be 3 ints in 0..255, got {self.color}")
        self.color = tuple(int(c) for c in self.color)
        self.position = (float(self.position[0]), float(self.position[1]))
        self.orientation = float(self.orientation) % 90.0

    def corners(self, width: float) -> list[tuple[float, float]]:
        """The 4 corners of the tile, counter-clockwise, for a given tile width."""
        cx, cy = self.position
        h = width / 2
        a = math.radians(self.orientation)
        cos_a, sin_a = math.cos(a), math.sin(a)
        return [
            (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
            for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))
        ]


@dataclass
class Mosaic:
    """A rectangular board of size width x height, origin at the bottom-left corner.

    tile_width : side length shared by every tile in this mosaic
    unit       : the length unit used for the mosaic AND all its tiles
                 (width, height, tile_width and positions), e.g. "cm", "mm", "in".
    """
    width: float
    height: float
    tile_width: float
    tiles: list[Tile] = field(default_factory=list)
    unit: str = "cm"

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Mosaic width and height must be > 0")
        if self.tile_width <= 0:
            raise ValueError(f"tile_width must be > 0, got {self.tile_width}")
        self.tile_width = float(self.tile_width)

    def add(self, tile: Tile) -> None:
        self.tiles.append(tile)

    def __len__(self) -> int:
        return len(self.tiles)
