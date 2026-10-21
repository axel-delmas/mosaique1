"""Sanity check for loss.py: start from a crowded, overlapping layout,
minimize the proximity loss with gradient descent, and save before/after images.

    python3 demo_loss.py
"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from loss import mosaic_to_tensors, naive_proximity_loss, tensors_to_mosaic
from mosaic import Mosaic, Tile
from visualize import draw_mosaic


def crowded_mosaic(n=120, seed=0) -> Mosaic:
    random.seed(seed)
    m = Mosaic(width=30, height=30, tile_width=1.8, unit="cm")
    for _ in range(n):
        # tiles thrown into the middle of the board, so many overlap
        x, y = random.gauss(15, 4), random.gauss(15, 4)
        color = (random.randint(40, 230), random.randint(40, 230), random.randint(40, 230))
        m.add(Tile(color=color, position=(x, y), orientation=random.uniform(0, 90)))
    return m


def main():
    before = crowded_mosaic()
    pos, _ = mosaic_to_tensors(before)
    opt = torch.optim.Adam([pos], lr=0.05)

    for step in range(301):
        opt.zero_grad()
        loss = naive_proximity_loss(pos, before.tile_width)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss.item():10.4f}")

    after = tensors_to_mosaic(before, pos)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_mosaic(before, ax=axes[0]); axes[0].set_title("Before")
    draw_mosaic(after, ax=axes[1]); axes[1].set_title("After minimizing proximity loss")
    fig.tight_layout()
    fig.savefig("demo_loss.png", dpi=110)
    print("saved demo_loss.png")


if __name__ == "__main__":
    main()
