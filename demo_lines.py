"""Guide-line mosaic of the (new) Degas, step by step, saved in lines/.

    python demo_lines.py [tile_cm]         default 0.4

1_contours   contours (SAUGE blur-3 coarse, cleaned) + distance field
2_lines      guide lines: echo rows (colored by row) + background rows; red = corners
3_stages     tiles colored by placement stage (corners first)
4_mosaic     the mosaic
5_zoom       zoom on the heads
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from guide_lines import build_line_mosaic
from loss import load_image
from visualize import draw_mosaic

BOARD_W = 30.0
EDGE = "edge_maps/new_sauge_vitl_coarse_blur3.npy"
ZOOM = (5.0, 24.0, 27.5, 39.5)
OUT = "lines"


def save(fig, name, dpi=110):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/{name}", flush=True)


def main():
    tile = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    tag = f"{tile:g}"
    os.makedirs(OUT, exist_ok=True)
    image = load_image("degas.jpg")
    edge = np.load(EDGE)
    t = time.time()
    m, st = build_line_mosaic(edge, image, BOARD_W, tile, return_steps=True)
    bh, sp = st["board_h"], st["spacing"]
    n_cor = sum(len(l["corners"]) for l in st["lines"])
    cover = len(m) * tile ** 2 / (BOARD_W * bh)
    print(f"{tag} cm: {len(st['lines'])} lines, {n_cor} corners, {st['candidates']} candidate "
          f"tiles, {len(m)} placed, skipped per stage {st['skipped'].tolist()}, "
          f"coverage {100 * cover:.0f}%  ({time.time() - t:.0f}s)", flush=True)
    ext = (0, BOARD_W, 0, bh)

    fig, ax = plt.subplots(figsize=(9, 12))
    im = ax.imshow(st["dist_cm"], extent=ext, cmap="magma_r", vmax=6 * sp)
    ys, xs = np.nonzero(st["skeleton"])
    ax.scatter((xs + 0.5) / st["ppc"], bh - (ys + 0.5) / st["ppc"], s=0.3, c="k")
    fig.colorbar(im, ax=ax, fraction=0.03, label="distance to nearest contour (cm)")
    ax.set_title("1. Contours (+ board edge) and distance to them"); ax.axis("off")
    save(fig, f"{tag}_1_contours.png")

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(image.numpy(), extent=ext, alpha=0.25)
    cmap = plt.get_cmap("viridis")
    rows = max(l["row"] for l in st["lines"])
    for l in st["lines"]:
        P = np.vstack([l["P"], l["P"][:1]]) if l["closed"] else l["P"]
        col = "0.45" if l["row"] == rows else cmap(l["row"] / max(rows - 1, 1))
        ax.plot(P[:, 0], P[:, 1], color=col, lw=0.7)
        if l["corners"]:
            c = l["P"][l["corners"]]
            ax.scatter(c[:, 0], c[:, 1], s=9, c="red", zorder=3)
    ax.set_xlim(0, BOARD_W); ax.set_ylim(0, bh)
    ax.set_title(f"2. Guide lines: echo rows (dark→light = row 1→{rows}), "
                 f"background rows (grey)\nred dots = corners kept (60°–120° turns), {n_cor} total")
    ax.axis("off")
    save(fig, f"{tag}_2_lines.png")

    fig, ax = plt.subplots(figsize=(9, 12))
    stage_col = ["red"] + [cmap(k / max(rows - 1, 1)) for k in range(rows)] + ["0.6"]
    polys = [t.corners(tile) for t in m.tiles]
    ax.add_collection(PolyCollection(polys, facecolors=[stage_col[s] for s in st["stages"]],
                                     edgecolors="none"))
    ax.set_xlim(0, BOARD_W); ax.set_ylim(0, bh); ax.set_aspect("equal")
    ax.set_title("3. Tiles by placement stage: red = corners (first), "
                 "then echo rows 1→3, grey = background"); ax.axis("off")
    save(fig, f"{tag}_3_stages.png")

    fig, ax = plt.subplots(figsize=(12, 16))
    draw_mosaic(m, ax=ax, grout_width=0.3 * tile)
    ax.set_title(f"4. Guide-line mosaic · tiles {tag} cm · {len(m):,} tiles · "
                 f"coverage {100 * cover:.0f}%", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, f"{tag}_4_mosaic.png", dpi=int(110 / min(1, tile / 0.4) ** 0.5))

    fig, ax = plt.subplots(figsize=(12, 7.6))
    draw_mosaic(m, ax=ax, grout_width=0.8 * tile)
    ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
    ax.set_title(f"5. Zoom · tiles {tag} cm")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, f"{tag}_5_zoom.png")


if __name__ == "__main__":
    main()
