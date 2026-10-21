"""Color pipeline on the guide-line mosaic, step by step, saved in colors/.

    python demo_colors.py [tile_cm] [k]        defaults 0.4, 16

1_smoothing     original vs edge-preserving smoothed image
2_steps_zoom    zoom on the heads: mean colors (before) -> smooth+median+saturation
                -> 16-color palette -> + outline contrast
3_palette       the palette, and which tiles the outline step changed
4_final         the full mosaic with the final colors
"""
import copy
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guide_lines import build_line_mosaic
from loss import load_image
from tile_colors import color_tiles, tile_raw_colors
from visualize import draw_mosaic

BOARD_W = 30.0
EDGE = "edge_maps/new_sauge_vitl_coarse_blur3.npy"
ZOOM = (5.0, 24.0, 27.5, 39.5)
OUT = "colors"


def recolor(m, rgb):
    m2 = copy.deepcopy(m)
    for t, c in zip(m2.tiles, rgb):
        t.color = tuple(int(round(v * 255)) for v in np.clip(c, 0, 1))
    return m2


def main():
    tile = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    tag = f"{tile:g}_k{k}"
    os.makedirs(OUT, exist_ok=True)
    image_t = load_image("degas.jpg")
    image = image_t.numpy()
    m, st = build_line_mosaic(np.load(EDGE), image_t, BOARD_W, tile, return_steps=True)
    bh = st["board_h"]
    C = np.array([t.position for t in m.tiles])
    A = np.radians([t.orientation for t in m.tiles])
    t0 = time.time()
    final, cs = color_tiles(image, C, A, st["rows"], tile, BOARD_W, bh, st["skeleton"],
                            st["ppc"], k=k, return_steps=True)
    print(f"{tag}: {len(m)} tiles, colors in {time.time() - t0:.1f}s, "
          f"outline tiles {cs['outline_mask'].sum()}, changed by contrast {cs['changed'].sum()}",
          flush=True)
    mean_before = tile_raw_colors(image, C, A, tile, BOARD_W, bh, stat="mean")

    ext = (0, BOARD_W, 0, bh)
    fig, axes = plt.subplots(1, 2, figsize=(14, 9.5))
    for ax, im, title in zip(axes, [image, cs["smoothed"]], ["original", "edge-preserving smoothing"]):
        ax.imshow(im, extent=ext); ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
        ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}_1_smoothing.png", dpi=100); plt.close(fig)

    versions = [(mean_before, "before: mean color"),
                (cs["boosted"], "smoothed + median + saturation ×1.2"),
                (cs["quantized"], f"+ palette of {k} colors"),
                (final, "+ outline contrast")]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11.8))
    for ax, (rgb, title) in zip(axes.flat, versions):
        draw_mosaic(recolor(m, rgb), ax=ax, grout_width=0.8 * tile)
        ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}_2_steps_zoom.png", dpi=100); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw=dict(width_ratios=[1, 2.2]))
    pal = cs["palette_rgb"]
    axes[0].imshow(pal[:, None, :].repeat(3, 1), aspect="auto")
    axes[0].set_title(f"palette ({k} colors, dark → light)"); axes[0].axis("off")
    hl = recolor(m, np.where(cs["changed"][:, None], final,
                             0.85 * np.ones(3) - 0.0 * final))
    draw_mosaic(hl, ax=axes[1], grout_width=0.3 * tile)
    axes[1].set_title(f"tiles changed by the outline-contrast step ({cs['changed'].sum()})")
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_xlabel(""); axes[1].set_ylabel("")
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}_3_palette.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 16))
    draw_mosaic(recolor(m, final), ax=ax, grout_width=0.3 * tile)
    ax.set_title(f"Guide lines + {k}-color palette + outline contrast · tiles {tile:g} cm", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}_4_final.png", dpi=int(110 / min(1, tile / 0.4) ** 0.5))
    plt.close(fig)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
