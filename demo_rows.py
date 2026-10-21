"""Rows on the background side of contours: 3 (before) vs 1, with final colors.

    python demo_rows.py [tile_cm] [k]        defaults 0.25, 16

Saves colors/<tile>_rows_compare.png (full) and <tile>_rows_zoom.png, plus
colors/<tile>_objects.png (which regions count as objects).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demo_colors import BOARD_W, EDGE, ZOOM, recolor
from guide_lines import build_line_mosaic
from loss import load_image
from tile_colors import color_tiles
from visualize import draw_mosaic


def build(edge, image_t, tile, k, outside):
    m, st = build_line_mosaic(edge, image_t, BOARD_W, tile, outside_rows=outside,
                              return_steps=True)
    C = np.array([t.position for t in m.tiles])
    A = np.radians([t.orientation for t in m.tiles])
    final = color_tiles(image_t.numpy(), C, A, st["rows"], tile, BOARD_W, st["board_h"],
                        st["skeleton"], st["ppc"], k=k)
    return recolor(m, final), st


def main():
    tile = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    image_t = load_image("degas.jpg")
    edge = np.load(EDGE)
    res = [(build(edge, image_t, tile, k, o), o) for o in (3, 1)]
    for (m, st), o in res:
        print(f"outside rows {o}: {len(m)} tiles", flush=True)
    st = res[1][0][1]

    fig, ax = plt.subplots(figsize=(8, 10.5))
    ax.imshow(image_t.numpy(), extent=(0, BOARD_W, 0, st["board_h"]))
    ax.imshow(np.where(st["objects"], 1.0, 0.0), extent=(0, BOARD_W, 0, st["board_h"]),
              cmap="bwr", alpha=0.35, vmin=0, vmax=1)
    ax.set_title("objects (red) = regions enclosed by contours\nbackground (blue) = regions touching the border")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(f"colors/{tile:g}_objects.png", dpi=80); plt.close(fig)

    for name, zoom, size, dpi in [("rows_compare", None, (22, 15.5), 90),
                                   ("rows_zoom", ZOOM, (22, 7.4), 100)]:
        fig, axes = plt.subplots(1, 2, figsize=size)
        for ax, ((m, _), o) in zip(axes, res):
            draw_mosaic(m, ax=ax, grout_width=(0.8 if zoom else 0.3) * tile)
            ax.set_title(f"{o} row{'s' if o > 1 else ''} on the background side of contours "
                         f"· {len(m):,} tiles", fontsize=13)
            if zoom:
                ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        fig.tight_layout(); fig.savefig(f"colors/{tile:g}_{name}.png", dpi=dpi); plt.close(fig)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
