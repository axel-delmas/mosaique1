"""Small tiles: gradient descent vs network, same scoring, plus zoomed crops.

    TILE_W=0.4 python compare_small.py

Needs small_tiles/deformed_gd.pkl (small_tiles.py) and
small_tiles/network/mosaicnet.pt (train_nn.py, START=deformed OUT=small_tiles/network).
Saves small_tiles/3_compare.png and small_tiles/4_zoom.png
"""
import math
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from loss import NeighbourList, mosaic_to_tensors, total_loss
from nn_mosaic import MosaicNet, to_mosaic
from train_nn import BOARD_W, CONTACT, TILE_W, forward, setup
from visualize import draw_mosaic


def scores(m, S):
    pos, ori, col = mosaic_to_tensors(m, requires_grad=False)
    pairs = NeighbourList(math.sqrt(2) * TILE_W, 0).pairs(pos)
    _, ov, co, al, nb = total_loss(pos, ori, col, S["image"], m, alpha=0.5, beta=0.3,
                                   field=S["field"], weighting="magnitude", delta=0.3,
                                   contact_factor=CONTACT, return_parts=True, pairs=pairs)
    return ov.item(), co.item(), al.item(), nb.item()


def main():
    S = setup("deformed")
    gd = pickle.load(open("small_tiles/deformed_gd.pkl", "rb"))
    net = MosaicNet(S["x"].shape[1])
    net.load_state_dict(torch.load("small_tiles/network/mosaicnet.pt"))
    with torch.no_grad():
        nn_m = to_mosaic(forward(net, S), BOARD_W, S["board_h"], TILE_W)
    panels = [(gd, "Deformed grid + gradient descent"), (nn_m, "Network (started from deformed grid)")]
    fig, axes = plt.subplots(1, 3, figsize=(24, 11))
    axes[0].imshow(S["image"].numpy(), extent=(0, BOARD_W, 0, S["board_h"]))
    axes[0].set_title("Original", fontsize=13); axes[0].axis("off")
    for ax, (m, title) in zip(axes[1:], panels):
        ov, co, al, nb = scores(m, S)
        print(title, len(m), round(ov, 5), round(co, 5), round(al, 4), round(nb, 4), flush=True)
        draw_mosaic(m, ax=ax, grout_width=0.25 * TILE_W)
        ax.set_title(f"{title}\n{len(m)} tiles of {TILE_W:g} cm · overlap {ov:.4f} · "
                     f"corresp {co:.4f} · align {al:.3f} · neighbour {nb:.3f}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig("small_tiles/3_compare.png", dpi=110)
    print("saved small_tiles/3_compare.png")

    # zoom on the heads, where the rows curve most
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    box = (5, 22, 26, 38)                       # x0, x1, y0, y1 in cm
    axes[0].imshow(S["image"].numpy(), extent=(0, BOARD_W, 0, S["board_h"]))
    axes[0].set_title("Original (zoom)")
    for ax, (m, title) in zip(axes[1:], panels):
        draw_mosaic(m, ax=ax, grout_width=0.5 * TILE_W)
        ax.set_title(title)
    for ax in axes:
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig("small_tiles/4_zoom.png", dpi=110)
    print("saved small_tiles/4_zoom.png")


if __name__ == "__main__":
    main()
