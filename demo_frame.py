"""Edge field WITH the board frame as a line: field pictures + mosaics.

    python demo_frame.py [coarse|middle]

Saves edge_pipeline/<gran>_9_frame_field.png and <gran>_10_frame_mosaics.png
"""
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_correspondence import CONTACT, grid_mosaic
from demo_edge_field import ticks
from demo_edge_grid import ALPHA, BOARD_W, TILE_W, fit
from edge_field import build_edge_field, echo as echo_fn
from loss import load_image
from visualize import draw_mosaic

ECHOS = [1.0, 3.0, np.inf]
BETA = 0.3


def main():
    gran = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    image = load_image("degas.jpg")
    img = image.numpy()
    edge = np.load(f"edge_maps/sauge_vitl_{gran}.npy")
    h, w = img.shape[:2]
    board_h = BOARD_W * h / w
    before = grid_mosaic(BOARD_W, board_h, TILE_W, pitch=CONTACT * TILE_W)

    fields, fig_f, axes_f = [], *plt.subplots(1, len(ECHOS), figsize=(5.2 * len(ECHOS), 7.5))
    for ax, e in zip(axes_f, ECHOS):
        field, st = build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=e, frame=True,
                                     return_steps=True)
        fields.append(field)
        c2a, s2a, _, wgt = echo_fn(st["pruned"], st["c2"], st["s2"], st["px_per_tile"], e)
        ax.imshow(img, alpha=0.3)
        ax.imshow(wgt, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
        ticks(ax, c2a, s2a, wgt, step=int(st["px_per_tile"]), length=st["px_per_tile"] * 0.8)
        ax.set_title(f"echo = {'∞' if np.isinf(e) else f'{e:g}'} tiles", fontsize=13)
        ax.axis("off")
    fig_f.suptitle("9. Direction field with the board frame added as a line", fontsize=14, y=1.02)
    fig_f.tight_layout()
    fig_f.savefig(f"edge_pipeline/{gran}_9_frame_field.png", dpi=90, bbox_inches="tight")
    print("saved field", flush=True)

    fig, axes = plt.subplots(1, len(ECHOS), figsize=(5.2 * len(ECHOS), 7.5))
    for ax, e, field in zip(axes, ECHOS, fields):
        t = time.time()
        after, (ov, co, al) = fit(before, image, torch.tensor(field, dtype=torch.float32),
                                  ALPHA, BETA)
        print(f"echo={e} overlap={ov:.5f} corresp={co:.5f} align={al:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
        draw_mosaic(after, ax=ax, grout_width=0.2)
        ax.set_title(f"echo = {'∞' if np.isinf(e) else f'{e:g}'} tiles   β = {BETA}\n"
                     f"overlap {ov:.4f} · corresp {co:.4f} · align {al:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(f"10. Mosaics with frame ({gran}, α = {ALPHA})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"edge_pipeline/{gran}_10_frame_mosaics.png", dpi=100)
    print("saved mosaics")


if __name__ == "__main__":
    main()
