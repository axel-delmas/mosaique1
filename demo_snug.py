"""Before / after snugging, same zoom, with final colors.

    python demo_snug.py [tile_cm]        default 0.25
Saves colors/<tile>_snug_zoom.png, <tile>_snug_full.png
"""
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demo_colors import BOARD_W, EDGE, ZOOM, recolor
from guide_lines import build_line_mosaic
from loss import load_image
from tile_colors import color_tiles
from visualize import draw_mosaic


def build(edge, image_t, tile, rounds):
    t = time.time()
    m, st = build_line_mosaic(edge, image_t, BOARD_W, tile, snug_rounds=rounds, return_steps=True)
    C = np.array([q.position for q in m.tiles]); A = np.radians([q.orientation for q in m.tiles])
    final = color_tiles(image_t.numpy(), C, A, st["rows"], tile, BOARD_W, st["board_h"],
                        st["skeleton"], st["ppc"])
    cover = len(m) * tile ** 2 / (BOARD_W * st["board_h"])
    print(f"snug rounds {rounds}: {len(m)} tiles, coverage {100 * cover:.1f}% ({time.time() - t:.0f}s)",
          flush=True)
    return recolor(m, final), cover


def main():
    tile = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    image_t = load_image("degas.jpg"); edge = np.load(EDGE)
    res = [(build(edge, image_t, tile, r), r) for r in (0, 8)]
    for name, zoom, size, gw in [("snug_zoom", ZOOM, (22, 7.4), 0.8), ("snug_full", None, (22, 15.5), 0.3)]:
        fig, axes = plt.subplots(1, 2, figsize=size)
        for ax, ((m, cover), r) in zip(axes, res):
            draw_mosaic(m, ax=ax, grout_width=gw * tile)
            ax.set_title(("before snugging" if r == 0 else "after snugging + refilling")
                         + f" · {len(m):,} tiles · coverage {100 * cover:.1f}%", fontsize=13)
            if zoom:
                ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        fig.tight_layout(); fig.savefig(f"colors/{tile:g}_{name}.png", dpi=100); plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
