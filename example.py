"""Build a small demo mosaic and visualize it."""
import math
import random
import sys

from mosaic import Mosaic, Tile
from visualize import show_mosaic


def demo_mosaic() -> Mosaic:
    random.seed(1)
    size, gap = 1.8, 0.2
    m = Mosaic(width=40, height=30, tile_width=size, unit="cm")
    step = size + gap
    cx, cy = m.width / 2, m.height / 2
    y = step / 2
    while y < m.height:
        x = step / 2
        while x < m.width:
            d = math.hypot(x - cx, y - cy)
            # concentric color bands, with a little jitter like real hand-laid tiles
            if d < 6:
                color = (230, 180, 40)
            elif d < 10:
                color = (200, 60, 40)
            elif d < 14:
                color = (40, 90, 160)
            else:
                color = (240, 235, 220)
            color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
            m.add(Tile(color=color,
                  position=(x + random.uniform(-0.1, 0.1), y + random.uniform(-0.1, 0.1)),
                  orientation=random.uniform(0, 8)))
            x += step
        y += step
    return m


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    show_mosaic(demo_mosaic(), save_to=out)
