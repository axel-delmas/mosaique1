"""Show every step of edge_field.build_edge_field as PNGs in edge_pipeline/.

    python demo_edge_field.py [coarse|middle]

Reads edge_maps/sauge_vitl_<granularity>.npy (SAUGE ViT-L output for degas.jpg).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edge_field import build_edge_field
from loss import load_image

OUT = "edge_pipeline"
BOARD_W, TILE_W = 30.0, 1.0
ECHOS = [0.25, 1.0, 3.0, np.inf]      # in tile widths


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def panel(title, figsize=(6, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig, ax


def ticks(ax, c2, s2, weight, step, length, color="k", min_weight=0.02):
    """Short segments ALONG the local line direction, every `step` pixels."""
    h, w = c2.shape
    psi = 0.5 * np.arctan2(s2, c2) + np.pi / 2          # across-line -> along-line
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            wt = weight[y, x]
            if wt < min_weight:
                continue
            dx, dy = np.cos(psi[y, x]) * length / 2, np.sin(psi[y, x]) * length / 2
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color,
                    lw=1.4, alpha=float(min(1, 0.15 + wt)), solid_capstyle="round")


def main():
    gran = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    edge = np.load(f"edge_maps/sauge_vitl_{gran}.npy")
    image = load_image("degas.jpg").numpy()
    _, st = build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=1.0, return_steps=True)
    p = f"{gran}_"
    ppt = st["px_per_tile"]

    fig, ax = panel(f"1. SAUGE ViT-L {gran} edge map (border cleared)")
    ax.imshow(1 - st["normalized"], cmap="gray", vmin=0, vmax=1)
    save(fig, p + "1_edge_map.png")

    fig, ax = panel(f"2. Threshold ≥ 30%  ({100 * st['mask'].mean():.0f}% of pixels)")
    ax.imshow(~st["mask"], cmap="gray")
    save(fig, p + "2_threshold.png")

    fig, ax = panel("3. Skeleton (small holes filled first)")
    ax.imshow(~ndilate(st["skeleton"]), cmap="gray")
    save(fig, p + "3_skeleton.png")

    removed = st["skeleton"] & ~st["pruned"]
    fig, ax = panel("4. Pruned: spurs < ¾ tile and pieces < 1 tile removed (red)")
    rgb = np.ones(st["skeleton"].shape + (3,))
    rgb[ndilate(st["pruned"])] = 0
    rgb[ndilate(removed)] = (0.85, 0.1, 0.1)
    ax.imshow(rgb)
    save(fig, p + "4_pruned.png")

    fig, ax = panel("5. Line direction (smoothed over ½ tile)")
    ax.imshow(image, alpha=0.35)
    ax.imshow(np.ma.masked_where(~ndilate(st["pruned"]), np.ones_like(st["pruned"], float)),
              cmap="gray", vmin=0, vmax=2, alpha=0.6)
    lw = np.where(ndilate(st["pruned"], 2), 1.0, 0.0)
    ticks(ax, st["c2"], st["s2"], lw, step=int(ppt / 2), length=ppt * 0.7, color="navy")
    save(fig, p + "5_line_direction.png")

    fig, ax = panel("6. Distance to nearest line (tile widths)")
    im = ax.imshow(st["distance"], cmap="magma_r")
    ax.contour(st["distance"], levels=[1, 2, 3, 4, 6, 8], colors="white", linewidths=0.6)
    fig.colorbar(im, ax=ax, fraction=0.035)
    save(fig, p + "6_distance.png")

    fig, axes = plt.subplots(1, len(ECHOS), figsize=(5.2 * len(ECHOS), 7.5))
    for ax, e in zip(axes, ECHOS):
        _, s = build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=e, return_steps=True)
        # re-derive the propagated doubled angle for drawing
        from edge_field import echo as _echo
        c2a, s2a, _, wgt = _echo(s["pruned"], s["c2"], s["s2"], ppt, e)
        ax.imshow(image, alpha=0.3)
        ax.imshow(wgt, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
        ticks(ax, c2a, s2a, wgt, step=int(ppt), length=ppt * 0.8)
        lab = "∞ (everywhere)" if np.isinf(e) else f"{e:g} tile" + ("s" if e != 1 else "")
        ax.set_title(f"echo = {lab}", fontsize=13)
        ax.axis("off")
    fig.suptitle("7. Propagated direction field: tick = tile alignment, "
                 "opacity/blue = weight exp(−distance/echo)", fontsize=14, y=1.03)
    fig.tight_layout()
    save(fig, p + "7_field_echo.png")


def ndilate(mask, it=1):
    """Thicken 1-px lines slightly so they are visible in the PNGs."""
    from scipy import ndimage
    return ndimage.binary_dilation(mask, iterations=it)


if __name__ == "__main__":
    main()
