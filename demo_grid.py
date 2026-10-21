"""Grid of mosaics of one image, varying alpha (rows) and beta (columns).

    python demo_grid.py [image]

total = (1 - alpha) * overlap + alpha * correspondence + beta * alignment
Every cell starts from the same grey grid and runs the same optimizer.
Saves demo_grid.png (the grid) and demo_gvf.png (the image + its GVF field).
"""
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_correspondence import CONTACT, grid_mosaic
from gvf import compute_gvf
from loss import load_image, mosaic_to_tensors, tensors_to_mosaic, total_loss
from visualize import draw_mosaic

ALPHAS = [0.2, 0.5, 0.8, 0.95]
BETAS = [0.0, 0.01, 0.1, 1.0]
STEPS = 300


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
                                      field=field, contact_factor=CONTACT, return_parts=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            col.clamp_(0, 1)
    return tensors_to_mosaic(before, pos, ori, col), (ov.item(), co.item(), al.item())


def save_gvf_figure(image, field, board_w, board_h, path="demo_gvf.png", step=8):
    h, w = field.shape[:2]
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    X = (xs + 0.5) / w * board_w
    Y = board_h - (ys + 0.5) / h * board_h
    g = field[::step, ::step].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(image.numpy(), extent=(0, board_w, 0, board_h))
    axes[0].quiver(X, Y, g[..., 0], g[..., 1], color="white", scale=12, width=0.003)
    axes[0].set_title("Image + GVF direction")
    axes[1].imshow(field.norm(dim=-1).numpy(), extent=(0, board_w, 0, board_h), cmap="gray")
    axes[1].set_title("GVF strength (white = strong)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "degas.jpg"
    image = load_image(path)
    h, w = image.shape[:2]
    board_w = 30.0
    board_h = board_w * h / w
    before = grid_mosaic(board_w, board_h, 1.0, pitch=CONTACT * 1.0)

    field = compute_gvf(image)
    save_gvf_figure(image, field, board_w, board_h)

    fig, axes = plt.subplots(len(ALPHAS), len(BETAS),
                             figsize=(4.2 * len(BETAS), 5.4 * len(ALPHAS)))
    for r, alpha in enumerate(ALPHAS):
        for c, beta in enumerate(BETAS):
            t = time.time()
            after, (ov, co, al) = fit(before, image, field, alpha, beta)
            print(f"alpha={alpha:<5} beta={beta:<5} overlap={ov:.5f} "
                  f"corresp={co:.5f} align={al:.3f}  ({time.time() - t:.0f}s)")
            ax = axes[r, c]
            draw_mosaic(after, ax=ax, grout_width=0.2)
            ax.set_title(f"α = {alpha}   β = {beta}\n"
                         f"overlap {ov:.4f} · corresp {co:.4f} · align {al:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("total = (1 − α)·overlap + α·correspondence + β·alignment", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig("demo_grid.png", dpi=110)
    print("saved demo_grid.png and demo_gvf.png")


if __name__ == "__main__":
    main()
