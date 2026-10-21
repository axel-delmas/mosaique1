"""Fit a mosaic to an image with total_loss = (1-alpha)*overlap + alpha*correspondence.

    python demo_correspondence.py [image] [alpha]

Starts from a regular grid of grey tiles and optimizes their colors, positions
and orientations together. Saves demo_correspondence.png (target / before / after).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from loss import load_image, mosaic_to_tensors, tensors_to_mosaic, total_loss
from mosaic import Mosaic, Tile
from visualize import draw_mosaic

CONTACT = 1.1   # tiles repel when centers are closer than 1.1 * tile_width


def grid_mosaic(width, height, tile_width, pitch) -> Mosaic:
    m = Mosaic(width, height, tile_width, unit="cm")
    y = pitch / 2
    while y < height:
        x = pitch / 2
        while x < width:
            m.add(Tile(color=(128, 128, 128), position=(x, y), orientation=0))
            x += pitch
        y += pitch
    return m


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "degas.jpg"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    image = load_image(path)
    h, w = image.shape[:2]

    board_w = 30.0
    board_h = board_w * h / w            # keep the image's aspect ratio
    tile_w = 1.0
    before = grid_mosaic(board_w, board_h, tile_w, pitch=CONTACT * tile_w)
    print(f"{len(before)} tiles on a {board_w:g} x {board_h:.1f} cm board, alpha = {alpha}")

    pos, ori, col = mosaic_to_tensors(before)
    col.requires_grad_()
    opt = torch.optim.Adam([
        {"params": [col], "lr": 0.05},
        {"params": [pos], "lr": 0.01},
        {"params": [ori], "lr": 0.02},
    ])

    for step in range(401):
        opt.zero_grad()
        loss, ov, co, _ = total_loss(pos, ori, col, image, before, alpha=alpha,
                                  contact_factor=CONTACT, return_parts=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            col.clamp_(0, 1)
        if step % 50 == 0:
            print(f"step {step:3d}  total {loss.item():.5f}  "
                  f"overlap {ov.item():.5f}  correspondence {co.item():.5f}")

    after = tensors_to_mosaic(before, pos, ori, col)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    axes[0].imshow(image.numpy(), extent=(0, board_w, 0, board_h))
    axes[0].set_title("Target image")
    draw_mosaic(before, ax=axes[1]); axes[1].set_title("Before")
    draw_mosaic(after, ax=axes[2]); axes[2].set_title(f"After (alpha = {alpha})")
    fig.tight_layout()
    fig.savefig("demo_correspondence.png", dpi=110)
    print("saved demo_correspondence.png")


if __name__ == "__main__":
    main()
