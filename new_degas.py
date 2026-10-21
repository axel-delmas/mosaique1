"""New (higher-res) Degas at several tile sizes: deformed grid + gradient descent.

    python new_degas.py [size_cm ...]          default: 0.4 0.25 0.15

Edges: SAUGE ViT-L coarse, run on the image blurred with σ = 3 px
(edge_maps/new_sauge_vitl_coarse_blur3.npy). Everything that shapes the edge
lines and the direction field is fixed in CENTIMETERS, so the lines are the same
at every tile size and only the tiles change.
Saves new_degas/<size>_full.png, <size>_zoom.png and new_degas/zoom_sizes.png
"""
import math
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deformed_grid import deformed_grid_mosaic
from edge_field import build_edge_field
from loss import NeighbourList, load_image, mosaic_to_tensors, tensors_to_mosaic, total_loss
from visualize import draw_mosaic

BOARD_W = 30.0
CONTACT = 1.1
ALPHA, BETA, DELTA = 0.5, 0.3, 0.3
EDGE = "edge_maps/new_sauge_vitl_coarse_blur3.npy"
# edge-line cleanup and field, in cm (converted to tile widths below)
SPUR_CM, MIN_LENGTH_CM, HOLE_CM2, SMOOTH_CM, ECHO_CM = 0.5, 1.0, 0.25, 0.3, 1.5
ZOOM = (5.0, 24.0, 27.5, 39.5)          # x0, x1, y0, y1 in cm: the two heads
OUT = "new_degas"


def fit_fast(before, image, field, tile_w, steps=300):
    pos, ori, col = mosaic_to_tensors(before)
    col.requires_grad_()
    opt = torch.optim.Adam([{"params": [col], "lr": 0.05},
                            {"params": [pos], "lr": 0.01 * tile_w},
                            {"params": [ori], "lr": 0.02}])
    nl = NeighbourList(math.sqrt(2) * tile_w, 0.25 * tile_w)
    for _ in range(steps):
        opt.zero_grad()
        parts = total_loss(pos, ori, col, image, before, alpha=ALPHA, beta=BETA,
                           field=field, weighting="magnitude", delta=DELTA,
                           contact_factor=CONTACT, return_parts=True, pairs=nl.pairs(pos))
        parts[0].backward()
        opt.step()
        with torch.no_grad():
            col.clamp_(0, 1)
    return tensors_to_mosaic(before, pos, ori, col), [p.item() for p in parts[1:]]


def run(size, image, edge, board_h):
    t0 = time.time()
    field_np = build_edge_field(edge, BOARD_W, size, echo_tiles=ECHO_CM / size,
                                spur_length=SPUR_CM / size, min_length=MIN_LENGTH_CM / size,
                                hole_area=HOLE_CM2 / size ** 2, smoothing=SMOOTH_CM / size,
                                border_px=20, frame=True)
    raw = deformed_grid_mosaic(field_np, image, BOARD_W, board_h, size, CONTACT * size)
    t1 = time.time()
    gd, (ov, co, al, nb) = fit_fast(raw, image, torch.tensor(field_np, dtype=torch.float32), size)
    t2 = time.time()
    print(f"{size:g} cm: {len(gd)} tiles | deformed grid {t1 - t0:.0f}s, gradient descent "
          f"{t2 - t1:.0f}s | overlap {ov:.4f} corresp {co:.4f} align {al:.3f} "
          f"neighbour {nb:.3f}", flush=True)
    label = f"tiles {size:g} cm · {len(gd):,} tiles"
    scores = f"overlap {ov:.4f} · corresp {co:.4f} · align {al:.3f} · neighbour {nb:.3f}"

    fig, ax = plt.subplots(figsize=(12, 16))
    draw_mosaic(gd, ax=ax, grout_width=0.3 * size)
    ax.set_title(f"{label}\n{scores}", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{size:g}_full.png", dpi=int(110 / min(1, size / 0.4) ** 0.5))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7.6))
    draw_mosaic(gd, ax=ax, grout_width=0.8 * size)
    ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
    ax.set_title(label, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{size:g}_zoom.png", dpi=110)
    plt.close(fig)
    pickle.dump(gd, open(f"{OUT}/{size:g}_mosaic.pkl", "wb"))
    print(f"saved {OUT}/{size:g}_full.png and _zoom.png", flush=True)
    return gd, label


def main():
    sizes = [float(s) for s in sys.argv[1:]] or [0.4, 0.25, 0.15]
    os.makedirs(OUT, exist_ok=True)
    image = load_image("degas.jpg")
    board_h = BOARD_W * image.shape[0] / image.shape[1]
    edge = np.load(EDGE)
    results = [run(s, image, edge, board_h) for s in sizes]

    fig, axes = plt.subplots(1, len(results) + 1, figsize=(6.5 * (len(results) + 1), 4.6))
    axes[0].imshow(image.numpy(), extent=(0, BOARD_W, 0, board_h))
    axes[0].set_title("original")
    for ax, (m, label) in zip(axes[1:], results):
        draw_mosaic(m, ax=ax, grout_width=0.8 * m.tile_width)
        ax.set_title(label)
    for ax in axes:
        ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{OUT}/zoom_sizes.png", dpi=110)
    print(f"saved {OUT}/zoom_sizes.png")


if __name__ == "__main__":
    main()
