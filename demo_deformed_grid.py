"""Deformed-grid pipeline on the Degas, step by step, saved in rows/.

    python demo_deformed_grid.py [coarse|middle]

2_orientation      smoothed 4-way orientation on the sample points
3_lattice          lattice votes; red = seams where neighbouring lattices disagree
4_raw              tiles straight from the deformed grid (mean image colors)
5_compare          regular grid + losses  vs  deformed grid raw  vs  deformed grid + losses
"""
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deformed_grid import (deformed_grid_mosaic, orientation_on_samples, sample_grid,
                           seam_mask, solve_position_field)
from demo_correspondence import CONTACT, grid_mosaic
from demo_neighbour import ECHO, fit
from edge_field import build_edge_field
from loss import load_image
from visualize import draw_mosaic

BOARD_W, TILE_W = 30.0, 1.0
SPACING = CONTACT * TILE_W          # tile + grout, same pitch as the regular grid
DELTA = 0.3


def save(fig, name):
    fig.savefig(f"rows/{name}", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("saved rows/" + name, flush=True)


def main():
    gran = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    image = load_image("degas.jpg")
    img = image.numpy()
    board_h = BOARD_W * img.shape[0] / img.shape[1]
    edge = np.load(f"edge_maps/sauge_vitl_{gran}.npy")
    field = build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=ECHO, frame=True)
    ext = (0, BOARD_W, 0, board_h)

    # 2. orientation
    X = sample_grid(BOARD_W, board_h, SPACING / 3)
    theta = orientation_on_samples(field, X, BOARD_W, board_h)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(img, extent=ext, alpha=0.35)
    for y in range(1, X.shape[0], 3):
        for x in range(1, X.shape[1], 3):
            p, t = X[y, x], theta[y, x]
            for a in (t, t + np.pi / 2):
                d = 0.35 * SPACING * np.array([np.cos(a), np.sin(a)])
                ax.plot(*np.stack([p - d, p + d]).T, color="navy", lw=1)
    ax.set_title("2. Orientation per sample (4-way, smoothed): little crosses = lattice axes")
    ax.axis("off")
    save(fig, f"2_orientation_{gran}.png")

    # 3. lattice
    t0 = time.time()
    Q = solve_position_field(X, theta, SPACING)
    print(f"position field: {time.time() - t0:.1f}s", flush=True)
    seam = seam_mask(Q, theta, SPACING)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(img, extent=ext, alpha=0.3)
    q = Q.reshape(-1, 2)
    ax.scatter(q[:, 0], q[:, 1], s=2, c="k", alpha=0.25)
    ax.scatter(X[seam][:, 0], X[seam][:, 1], s=3, c="red", alpha=0.6)
    ax.set_xlim(0, BOARD_W); ax.set_ylim(0, board_h)
    ax.set_title(f"3. Lattice votes (black dots pile up on tile spots)\n"
                 f"red = seams (lattices off by > ¼ tile), {100 * seam.mean():.0f}% of samples")
    ax.axis("off")
    save(fig, f"3_lattice_{gran}.png")

    # 4. raw mosaic
    raw = deformed_grid_mosaic(field, image, BOARD_W, board_h, TILE_W, SPACING)
    fig, ax = plt.subplots(figsize=(7, 9))
    draw_mosaic(raw, ax=ax, grout_width=0.2)
    ax.set_title(f"4. Straight from the deformed grid: {len(raw)} tiles")
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, f"4_raw_{gran}.png")

    # 5. comparison
    grid0 = grid_mosaic(BOARD_W, board_h, TILE_W, pitch=SPACING)
    fld = torch.tensor(field, dtype=torch.float32)
    t0 = time.time()
    from_grid, pg = fit(grid0, image, fld, DELTA)
    polished, pd = fit(raw, image, fld, DELTA)
    print(f"polish: {time.time() - t0:.0f}s", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.8))
    for ax, (m, title) in zip(axes, [
            (from_grid, f"Regular grid + losses (δ = {DELTA})\n"
                        f"overlap {pg[0]:.4f} · corresp {pg[1]:.4f} · align {pg[2]:.2f} · neighbour {pg[3]:.2f}"),
            (raw, f"Deformed grid, raw ({len(raw)} tiles)"),
            (polished, f"Deformed grid + losses (δ = {DELTA})\n"
                       f"overlap {pd[0]:.4f} · corresp {pd[1]:.4f} · align {pd[2]:.2f} · neighbour {pd[3]:.2f}")]):
        draw_mosaic(m, ax=ax, grout_width=0.2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(f"5. Regular vs deformed grid start ({gran}, frame, echo {ECHO:g})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"5_compare_{gran}.png")


if __name__ == "__main__":
    main()
