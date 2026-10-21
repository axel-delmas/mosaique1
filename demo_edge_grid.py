"""Grid of Degas mosaics aligned to the SAUGE edge field: echo (rows) x beta (columns).

    python demo_edge_grid.py [coarse|middle]

total = (1 - alpha) * overlap + alpha * correspondence + beta * alignment,
with alpha fixed at ALPHA and the alignment field from edge_field.build_edge_field.
Saves edge_pipeline/<granularity>_8_mosaic_grid.png
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

ALPHA = 0.5
ECHOS = [0.25, 1.0, 3.0, np.inf]     # tile widths
BETAS = [0.1, 0.3, 1.0]
STEPS = 300
BOARD_W, TILE_W = 30.0, 1.0


def fit(before, image, field, alpha, beta, steps=STEPS):
    pos, ori, col = mosaic_to_tensors(before)
    col.requires_grad_()
    opt = torch.optim.Adam([
        {"params": [col], "lr": 0.05},
        {"params": [pos], "lr": 0.01},
        {"params": [ori], "lr": 0.02},
    ])
    for _ in range(steps):
        opt.zero_grad()
        loss, ov, co, al = total_loss(pos, ori, col, image, before, alpha=alpha, beta=beta,
                                      field=field, weighting="magnitude",
                                      contact_factor=CONTACT, return_parts=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            col.clamp_(0, 1)
    return tensors_to_mosaic(before, pos, ori, col), (ov.item(), co.item(), al.item())


def main():
    gran = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    image = load_image("degas.jpg")
    edge = np.load(f"edge_maps/sauge_vitl_{gran}.npy")
    h, w = image.shape[:2]
    board_h = BOARD_W * h / w
    before = grid_mosaic(BOARD_W, board_h, TILE_W, pitch=CONTACT * TILE_W)

    fig, axes = plt.subplots(len(ECHOS), len(BETAS),
                             figsize=(4.4 * len(BETAS), 5.6 * len(ECHOS)))
    for r, e in enumerate(ECHOS):
        field = torch.tensor(build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=e),
                             dtype=torch.float32)
        for c, beta in enumerate(BETAS):
            t = time.time()
            after, (ov, co, al) = fit(before, image, field, ALPHA, beta)
            lab = "∞" if np.isinf(e) else f"{e:g}"
            print(f"{gran} echo={lab:<5} beta={beta:<4} overlap={ov:.5f} "
                  f"corresp={co:.5f} align={al:.3f} ({time.time() - t:.0f}s)", flush=True)
            ax = axes[r, c]
            draw_mosaic(after, ax=ax, grout_width=0.2)
            ax.set_title(f"echo = {lab} tiles   β = {beta}\n"
                         f"overlap {ov:.4f} · corresp {co:.4f} · align {al:.2f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(f"SAUGE ViT-L {gran} edge field  ·  α = {ALPHA}  ·  "
                 "total = (1−α)·overlap + α·corresp + β·align", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs("edge_pipeline", exist_ok=True)
    out = f"edge_pipeline/{gran}_8_mosaic_grid.png"
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
