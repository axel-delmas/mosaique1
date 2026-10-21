"""Step 1 of the guided pipeline: automatic segmentation with SAM, turned into
a numbered partition of the image for the user to name and group.

    python pipeline_step1_segment.py <sam_checkpoint> [points_per_side]

Writes pipeline/step1_regions.npy  (H, W) int, 0 = unassigned, 1..N = regions
       pipeline/step1_regions.png  numbered overlay for the user
       pipeline/step1_masks.npz    raw SAM masks (for later refinement)
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "/tmp/edge/SAUGE")                 # segment_anything package
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

OUT = "pipeline"
MIN_AREA = 0.002                                      # regions below 0.2% of the image are merged


def partition(masks, shape, min_area):
    """Paint masks largest first so smaller (more detailed) masks win, then
    relabel connected pieces and absorb tiny ones into their largest neighbour."""
    lab = np.zeros(shape, int)
    for k, m in enumerate(sorted(masks, key=lambda m: -m["area"]), 1):
        lab[m["segmentation"]] = k
    # connected components of each label
    out = np.zeros(shape, int)
    n = 0
    for k in np.unique(lab):
        cc, c = ndimage.label(lab == k)
        out[cc > 0] = cc[cc > 0] + n
        n += c
    # absorb small pieces (and unassigned 0-pixels that are small) into neighbours
    min_px = min_area * out.size
    for _ in range(5):
        sizes = np.bincount(out.ravel(), minlength=n + 1)
        small = sizes < min_px
        small[0] = True
        if not small[out].any():
            break
        keep = ~small[out]
        _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
        out = np.where(keep, out, out[iy, ix])
    # renumber 1..N, largest first
    ids, counts = np.unique(out, return_counts=True)
    order = ids[np.argsort(-counts)]
    remap = np.zeros(out.max() + 1, int)
    remap[order] = np.arange(1, len(order) + 1)
    return remap[out]


def overlay(image, regions, path):
    n = regions.max()
    rng = np.random.default_rng(3)
    colors = rng.uniform(0.15, 1.0, (n + 1, 3))
    fig, ax = plt.subplots(figsize=(11, 11 * image.shape[0] / image.shape[1]))
    ax.imshow(image)
    ax.imshow(colors[regions], alpha=0.45)
    edges = np.zeros(regions.shape, bool)
    edges[:-1] |= regions[:-1] != regions[1:]
    edges[:, :-1] |= regions[:, :-1] != regions[:, 1:]
    ax.imshow(np.ma.masked_where(~edges, edges), cmap="gray_r", vmin=0, vmax=1)
    for k in range(1, n + 1):
        d = ndimage.distance_transform_edt(np.pad(regions == k, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(d.argmax(), d.shape)
        ax.text(x, y, str(k), ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.7))
    ax.set_title(f"Step 1 · {n} regions found by SAM — tell me which numbers form which object")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ckpt = sys.argv[1]
    pps = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    os.makedirs(OUT, exist_ok=True)
    image = np.asarray(Image.open("degas.jpg").convert("RGB"))
    kind = "vit_l" if "_l" in os.path.basename(ckpt) else "vit_b"
    sam = sam_model_registry[kind](checkpoint=ckpt)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=pps, pred_iou_thresh=0.86,
                                    stability_score_thresh=0.90, min_mask_region_area=500)
    t = time.time()
    with torch.no_grad():
        masks = gen.generate(image)
    print(f"SAM {kind}: {len(masks)} masks in {time.time() - t:.0f}s", flush=True)
    np.savez_compressed(f"{OUT}/step1_masks.npz",
                        masks=np.array([m["segmentation"] for m in masks]),
                        areas=np.array([m["area"] for m in masks]))
    regions = partition(masks, image.shape[:2], MIN_AREA)
    np.save(f"{OUT}/step1_regions.npy", regions)
    overlay(image, regions, f"{OUT}/step1_regions.png")
    print(f"{regions.max()} regions -> {OUT}/step1_regions.png", flush=True)


if __name__ == "__main__":
    main()
