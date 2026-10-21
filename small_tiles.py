"""The whole pipeline with much smaller tiles (default 0.4 cm instead of 1 cm).

    TILE_W=0.4 python small_tiles.py

1. edge field (SAUGE coarse, frame, echo 3 tiles)
2. deformed grid                      -> small_tiles/1_deformed_raw.png
3. deformed grid + gradient descent   -> small_tiles/2_deformed_gd.png
   (overlap / neighbour terms use a NeighbourList: only nearby pairs)
4. comparison with the 1 cm result    -> small_tiles/3_compare_sizes.png
The network version is trained separately:
    TILE_W=0.4 START=deformed OUT=small_tiles/network python train_nn.py 1000
"""
import math
import os
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
TILE_W = float(os.environ.get("TILE_W", 0.4))
CONTACT = 1.1
SPACING = CONTACT * TILE_W
ALPHA, BETA, DELTA = 0.5, 0.3, 0.3
OUT = "small_tiles"


def fit_fast(before, image, field, steps=300):
    """Same optimizer as before, lrs scaled to the tile size, pairs from a NeighbourList."""
    pos, ori, col = mosaic_to_tensors(before)
    col.requires_grad_()
    opt = torch.optim.Adam([{"params": [col], "lr": 0.05},
                            {"params": [pos], "lr": 0.01 * TILE_W},
                            {"params": [ori], "lr": 0.02}])
    nl = NeighbourList(math.sqrt(2) * TILE_W, 0.25 * TILE_W)
    for _ in range(steps):
        opt.zero_grad()
        parts = total_loss(pos, ori, col, image, before, alpha=ALPHA, beta=BETA,
                           field=field, weighting="magnitude", delta=DELTA,
                           contact_factor=CONTACT, return_parts=True, pairs=nl.pairs(pos))
        parts[0].backward()
        opt.step()
        with torch.no_grad():
            col.clamp_(0, 1)
    return tensors_to_mosaic(before, pos, ori, col), [p.item() for p in parts[1:]], nl.rebuilds


def show(m, title, path, figsize=(10, 13), dpi=150):
    fig, ax = plt.subplots(figsize=figsize)
    draw_mosaic(m, ax=ax, grout_width=0.25 * TILE_W)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("saved", path, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    image = load_image("degas.jpg")
    board_h = BOARD_W * image.shape[0] / image.shape[1]
    edge = np.load("edge_maps/sauge_vitl_coarse.npy")
    t = time.time()
    field_np = build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=3.0, frame=True)
    field = torch.tensor(field_np, dtype=torch.float32)
    print(f"edge field {time.time() - t:.1f}s", flush=True)

    t = time.time()
    raw = deformed_grid_mosaic(field_np, image, BOARD_W, board_h, TILE_W, SPACING)
    print(f"deformed grid: {len(raw)} tiles, {time.time() - t:.1f}s", flush=True)
    show(raw, f"Deformed grid, tiles {TILE_W:g} cm: {len(raw)} tiles", f"{OUT}/1_deformed_raw.png")

    t = time.time()
    gd, (ov, co, al, nb), rebuilds = fit_fast(raw, image, field)
    print(f"gradient descent: {time.time() - t:.0f}s, neighbour list rebuilt {rebuilds}x "
          f"overlap {ov:.4f} corresp {co:.4f} align {al:.3f} neighbour {nb:.3f}", flush=True)
    show(gd, f"Deformed grid + gradient descent, tiles {TILE_W:g} cm: {len(gd)} tiles\n"
             f"overlap {ov:.4f} · corresp {co:.4f} · align {al:.3f} · neighbour {nb:.3f}",
         f"{OUT}/2_deformed_gd.png")
    import pickle
    pickle.dump(gd, open(f"{OUT}/deformed_gd.pkl", "wb"))


if __name__ == "__main__":
    main()
