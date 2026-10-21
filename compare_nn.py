"""Side-by-side: gradient descent vs the network, both starts.

    python compare_nn.py

Needs network/from_grid/mosaicnet.pt and network/from_deformed/mosaicnet.pt
(from train_nn.py with START=grid / START=deformed). Saves network/compare.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from demo_neighbour import fit
from deformed_grid import deformed_grid_mosaic
from loss import mosaic_to_tensors, total_loss
from train_nn import BOARD_W, CONTACT, SPACING, TILE_W, forward, setup
from nn_mosaic import MosaicNet, to_mosaic
from visualize import draw_mosaic


def scores(m, S):
    """Same four loss terms for every mosaic, computed the same way."""
    pos, ori, col = mosaic_to_tensors(m, requires_grad=False)
    _, ov, co, al, nb = total_loss(pos, ori, col, S["image"], m, alpha=0.5, beta=0.3,
                                   field=S["field"], weighting="magnitude", delta=0.3,
                                   contact_factor=CONTACT, return_parts=True)
    return ov.item(), co.item(), al.item(), nb.item()


def main():
    os.makedirs("network", exist_ok=True)
    S = setup("grid")
    dg = deformed_grid_mosaic(S["field_np"], S["image"], BOARD_W, S["board_h"], TILE_W, SPACING)
    dg_gd, _ = fit(dg, S["image"], S["field"], 0.3)
    panels = [(dg, "Deformed grid (algorithm)"), (dg_gd, "Deformed grid + gradient descent")]
    for start in ("grid", "deformed"):
        Ss = setup(start)
        net = MosaicNet(Ss["x"].shape[1])
        net.load_state_dict(torch.load(f"network/from_{start}/mosaicnet.pt"))
        with torch.no_grad():
            out = forward(net, Ss)
        m = to_mosaic(out, BOARD_W, Ss["board_h"], TILE_W)
        panels.append((m, f"Network, started from {'regular' if start == 'grid' else 'deformed'} grid"))

    fig, axes = plt.subplots(1, 4, figsize=(21, 8.2))
    for ax, (m, title) in zip(axes, panels):
        ov, co, al, nb = scores(m, S)
        draw_mosaic(m, ax=ax, grout_width=0.2)
        ax.set_title(f"{title}\n{len(m)} tiles · overlap {ov:.4f} · corresp {co:.4f}\n"
                     f"align {al:.3f} · neighbour {nb:.3f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        print(title, len(m), round(ov, 5), round(co, 5), round(al, 4), round(nb, 4), flush=True)
    fig.suptitle("Gradient descent vs network (losses: α 0.5, β 0.3, δ 0.3; "
                 "scores recomputed identically for all four)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("network/compare.png", dpi=95)
    print("saved network/compare.png")


if __name__ == "__main__":
    main()
