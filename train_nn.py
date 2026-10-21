"""Train MosaicNet on one image (the Degas) with the mosaic losses.

    START=deformed python train_nn.py [steps]     (START=grid for the regular grid)

total = (1-α)·overlap + α·corresp + β·align + δ·neighbour
        + γ·coverage + λ(t)·decisiveness + μ·boundary

Snapshots (tiles with p > 0.5 + presence map) go to network/ as it trains,
plus a loss-curve plot and the final comparison.
"""
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_correspondence import CONTACT
from edge_field import build_edge_field
from loss import load_image
from nn_mosaic import (MosaicNet, boundary_loss, decode, make_input,
                       pair_losses, tile_losses, to_mosaic)
from visualize import draw_mosaic

BOARD_W = 30.0
TILE_W = float(os.environ.get("TILE_W", 1.0))
SPACING = CONTACT * TILE_W
CELL = SPACING / 2
WINDOW = 4
ALPHA, BETA, DELTA = 0.5, 0.3, 0.3
GAMMA = float(os.environ.get("GAMMA", 1.0))     # coverage
LAMBDA = float(os.environ.get("LAMBDA", 0.5))   # decisiveness (final value)
MU = 1.0                                        # boundary
PRIOR = float(os.environ.get("PRIOR", 2.0))     # strength of the starting layout
START = os.environ.get("START", "grid")         # "grid" or "deformed" (deformed_grid.py result)
OUT = os.environ.get("OUT", f"network/from_{START}")


def snapshot(out, step, board_h, parts, path):
    m = to_mosaic(out, BOARD_W, board_h, TILE_W)
    p = torch.sigmoid(out["logit"]).detach().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), gridspec_kw=dict(width_ratios=[1, 1]))
    draw_mosaic(m, ax=axes[0], grout_width=0.2 * min(1.0, TILE_W))
    axes[0].set_title(f"step {step}: {len(m)} tiles (p > 0.5)\n{parts}", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].set_xlabel(""); axes[0].set_ylabel("")
    im = axes[1].imshow(p, cmap="viridis", vmin=0, vmax=1, extent=(0, BOARD_W, 0, board_h))
    axes[1].set_title("presence p per cell"); axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=int(90 / min(1.0, TILE_W) ** 0.5))
    plt.close(fig)
    return m


def setup(start=START):
    """Everything fixed during training: image, field, cell grid, network input,
    and the starting layout (prior presence / offsets / angles)."""
    image = load_image("degas.jpg")
    board_h = BOARD_W * image.shape[0] / image.shape[1]
    edge = np.load("edge_maps/sauge_vitl_coarse.npy")
    field_np = build_edge_field(edge, BOARD_W, TILE_W, echo_tiles=3.0, frame=True)
    field = torch.tensor(field_np, dtype=torch.float32)
    gh, gw = int(round(board_h / CELL)), int(round(BOARD_W / CELL))
    cell_y = board_h / gh
    x = make_input(image, field_np, gh, gw)

    prior_off = torch.zeros(gh, gw, 2)
    prior_theta = torch.zeros(gh, gw)
    if start == "grid":
        # start from the regular grid: one cell in every 2x2 block is "on"
        prior = torch.full((gh, gw), -PRIOR)
        prior[::2, ::2] = PRIOR
    else:
        # start from the deformed grid: its tiles switch their cell "on", with the
        # exact position and angle as fixed offsets the network learns corrections to
        from deformed_grid import deformed_grid_mosaic
        dg = deformed_grid_mosaic(field_np, image, BOARD_W, board_h, TILE_W, SPACING)
        prior = torch.full((gh, gw), -PRIOR)
        for t in dg.tiles:
            tx, ty = t.position
            iy = min(gh - 1, max(0, int((board_h - ty) / cell_y)))
            ix = min(gw - 1, max(0, int(tx / CELL)))
            if prior[iy, ix] > 0:
                continue                      # two tiles in one cell: keep the first
            prior[iy, ix] = PRIOR
            cx, cy = (ix + 0.5) * CELL, board_h - (iy + 0.5) * cell_y
            prior_off[iy, ix] = torch.tensor([tx - cx, ty - cy])
            prior_theta[iy, ix] = math.radians(t.orientation)
        print(f"deformed grid start: {len(dg)} tiles, {int((prior > 0).sum())} placed", flush=True)

    return dict(image=image, board_h=board_h, field_np=field_np, field=field, gh=gh, gw=gw,
                cell_y=cell_y, x=x, prior=prior, prior_off=prior_off, prior_theta=prior_theta)


def forward(net, S):
    """Network output decoded into tile tensors, starting layout applied."""
    out = decode(net(S["x"]), CELL, S["board_h"])
    out["logit"] = 6 * torch.tanh(out["logit"] / 6) + S["prior"]   # bounded: never fully dead
    out["theta"] = out["theta"] + S["prior_theta"]
    # rows of the cell grid are cell_y tall (board_h may not be a multiple of CELL)
    out["pos"] = out["pos"] * torch.tensor([1.0, S["cell_y"] / CELL]) + \
        torch.tensor([0.0, S["board_h"] * (1 - S["cell_y"] / CELL)]) + S["prior_off"]
    return out


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    S = setup()
    image, board_h, field = S["image"], S["board_h"], S["field"]
    net = MosaicNet(S["x"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=2e-4)
    print(f"cells {S['gh']}x{S['gw']} = {S['gh'] * S['gw']}, "
          f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)

    snaps = {0, 100, 300, steps // 2, steps}
    hist = []
    t0 = time.time()
    for step in range(steps + 1):
        frac = step / steps
        lam = LAMBDA * min(1.0, max(0.0, (frac - 0.2) / 0.5))   # 0 until 20%, full at 70%
        out = forward(net, S)
        p = torch.sigmoid(out["logit"])
        z = p
        n_ref = BOARD_W * board_h / SPACING ** 2      # tiles the board holds at nominal spacing

        ov, nb = pair_losses(out["pos"], out["theta"], z, TILE_W, WINDOW, CONTACT, 4.0,
                             norm=n_ref)
        co, al = tile_losses(out["pos"], out["theta"], out["color"], z, image, field,
                             TILE_W, BOARD_W, board_h)
        cov = 1 - z.sum() / n_ref
        dec = (p * (1 - p)).mean()
        bd = boundary_loss(out["pos"], z, TILE_W, BOARD_W, board_h)
        total = ((1 - ALPHA) * ov + ALPHA * co + BETA * al + DELTA * nb
                 + GAMMA * cov + lam * dec + MU * bd)
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        n_tiles = int((p > 0.5).sum())
        hist.append([step, total.item(), ov.item(), co.item(), al.item(), nb.item(),
                     cov.item(), dec.item(), n_tiles])
        if step % 50 == 0:
            print(f"step {step:4d} lam {lam:.2f} total {total.item():.4f} ov {ov.item():.4f} "
                  f"co {co.item():.4f} al {al.item():.3f} nb {nb.item():.3f} "
                  f"cov {cov.item():.3f} dec {dec.item():.3f} tiles {n_tiles} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step in snaps:
            parts = (f"overlap {ov.item():.4f} · corresp {co.item():.4f} · "
                     f"align {al.item():.2f} · neighbour {nb.item():.2f}")
            snapshot(out, step, board_h, parts, f"{OUT}/step_{step:04d}.png")

    torch.save(net.state_dict(), f"{OUT}/mosaicnet.pt")
    np.save(f"{OUT}/history.npy", np.array(hist))

    h = np.array(hist)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, name in [(2, "overlap"), (3, "corresp"), (4, "align"), (5, "neighbour")]:
        axes[0].plot(h[:, 0], h[:, k], label=name)
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].set_title("loss terms")
    axes[1].plot(h[:, 0], h[:, 6], label="coverage (1 − density)")
    axes[1].plot(h[:, 0], h[:, 7], label="decisiveness p(1−p)")
    axes[1].legend(); axes[1].set_title("coverage / decisiveness")
    axes[2].plot(h[:, 0], h[:, 8]); axes[2].set_title("tiles (p > 0.5)")
    for ax in axes: ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(f"{OUT}/training_curves.png", dpi=90)
    print("done", flush=True)


if __name__ == "__main__":
    main()
