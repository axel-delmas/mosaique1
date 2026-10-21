"""Objects filled with echo rows + brick-bond background, with final colors.

    python demo_fill.py [tile_cm] [k]        defaults 0.25, 16
Saves colors/<tile>_fill_full.png, _fill_zoom.png, _fill_lines.png
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


def main():
    tile = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    image_t = load_image("degas.jpg")
    t = time.time()
    m, st = build_line_mosaic(np.load(EDGE), image_t, BOARD_W, tile, return_steps=True)
    C = np.array([t_.position for t_ in m.tiles])
    A = np.radians([t_.orientation for t_ in m.tiles])
    final = color_tiles(image_t.numpy(), C, A, st["rows"], tile, BOARD_W, st["board_h"],
                        st["skeleton"], st["ppc"], k=k)
    m = recolor(m, final)
    cover = len(m) * tile ** 2 / (BOARD_W * st["board_h"])
    print(f"{tile:g} cm: {len(st['lines'])} lines, {len(m)} tiles, coverage {100 * cover:.0f}%, "
          f"skipped per stage {st['skipped'].tolist()} ({time.time() - t:.0f}s)", flush=True)

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(image_t.numpy(), extent=(0, BOARD_W, 0, st["board_h"]), alpha=0.25)
    K = st["echo_rows"]
    cmap = plt.get_cmap("viridis")
    for l in st["lines"]:
        P = np.vstack([l["P"], l["P"][:1]]) if l["closed"] else l["P"]
        col = ("red" if l.get("kind") == "seam" else "0.5" if l.get("kind") == "background"
               else cmap(min(l["row"], 12) / 12))
        ax.plot(P[:, 0], P[:, 1], lw=0.8 if l.get("kind") == "seam" else 0.5, color=col)
    ax.set_xlim(0, BOARD_W); ax.set_ylim(0, st["board_h"]); ax.axis("off")
    ax.set_title("guide lines: contour rows (colored by row), background (grey)")
    fig.tight_layout(); fig.savefig(f"colors/{tile:g}_fill_lines.png", dpi=100); plt.close(fig)

    for name, zoom, size, dpi, gw in [("fill_full", None, (12, 16), 110, 0.3),
                                      ("fill_zoom", ZOOM, (13, 8.2), 110, 0.8)]:
        fig, ax = plt.subplots(figsize=size)
        draw_mosaic(m, ax=ax, grout_width=gw * tile)
        if zoom:
            ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
        ax.set_title(f"outside-in, whole lines first, no isolated tiles · "
                     f"tiles {tile:g} cm · {len(m):,} tiles", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        fig.tight_layout(); fig.savefig(f"colors/{tile:g}_{name}.png", dpi=dpi); plt.close(fig)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
