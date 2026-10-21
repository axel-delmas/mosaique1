"""Effect of the neighbour loss (delta), starting from the regular grid.

    python demo_neighbour.py [coarse|middle]

Edge field: SAUGE ViT-L, frame on, echo = 3 tiles. alpha = 0.5, beta = 0.3.
Saves rows/1_neighbour_loss_<gran>.png
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_correspondence import CONTACT, grid_mosaic
from edge_field import build_edge_field
from loss import load_image, mosaic_to_tensors, tensors_to_mosaic, total_loss
from visualize import draw_mosaic

ALPHA, BETA, ECHO = 0.5, 0.3, 3.0
DELTAS = [0.0, 0.3, 1.0, 3.0]
BOARD_W, TILE_W = 30.0, 1.0


def fit(before, image, field, delta, steps=300, pos=None, ori=None, col=None):
    if pos is None:
        pos, ori, col = mosaic_to_tensors(before)
    col = col.clone().requires_grad_()
    opt = torch.optim.Adam([{"params": [col], "lr": 0.05},
                            {"params": [pos], "lr": 0.01},
                            {"params": [ori], "lr": 0.02}])
    for _ in range(steps):
        opt.zero_grad()
        parts = total_loss(pos, ori, col, image, before, alpha=ALPHA, beta=BETA,
                           field=field, weighting="magnitude", delta=delta,
                           contact_factor=CONTACT, return_parts=True)
        parts[0].backward()
        opt.step()
        with torch.no_grad():
            col.clamp_(0, 1)
    return tensors_to_mosaic(before, pos, ori, col), [p.item() for p in parts[1:]]


def main():
    gran = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    image = load_image("degas.jpg")
    edge = np.load(f"edge_maps/sauge_vitl_{gran}.npy")
    board_h = BOARD_W * image.shape[0] / image.shape[1]
    field = torch.tensor(build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=ECHO, frame=True),
                         dtype=torch.float32)
    before = grid_mosaic(BOARD_W, board_h, TILE_W, pitch=CONTACT * TILE_W)
    fig, axes = plt.subplots(1, len(DELTAS), figsize=(5 * len(DELTAS), 7.6))
    for ax, delta in zip(axes, DELTAS):
        t = time.time()
        after, (ov, co, al, nb) = fit(before, image, field, delta)
        print(f"delta={delta} overlap={ov:.5f} corresp={co:.5f} align={al:.3f} "
              f"neighbour={nb:.3f} ({time.time() - t:.0f}s)", flush=True)
        draw_mosaic(after, ax=ax, grout_width=0.2)
        ax.set_title(f"δ = {delta}\noverlap {ov:.4f} · corresp {co:.4f}\n"
                     f"align {al:.2f} · neighbour {nb:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(f"Neighbour loss, from the regular grid  ({gran}, frame, echo {ECHO:g}, "
                 f"α {ALPHA}, β {BETA})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs("rows", exist_ok=True)
    fig.savefig(f"rows/1_neighbour_loss_{gran}.png", dpi=100)
    print("saved")


if __name__ == "__main__":
    main()
