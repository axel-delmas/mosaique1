"""Step 1: segmentation with Segment Anything (SAM), and automatic grouping.

The SAM weights are downloaded once into ~/.cache/mosaic_studio.
"""
from __future__ import annotations

import math
import os
import urllib.request

import numpy as np
from scipy import ndimage

from . import groups as G
from .project import new_group

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "mosaic_studio")
WEIGHTS = {   # official URL first, a mirror second
    "vit_b": ["https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
              "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt"],
    "vit_l": ["https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
              "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_l.pt"],
}
_models = {}
_predictor_cache = {}


def no_progress(frac, stage):
    """Default progress callback: progress(fraction in [0, 1], stage text)."""


def scaled(progress, a, b):
    """Progress callback mapping a sub-step's [0, 1] onto [a, b] of the whole."""
    return lambda frac, stage: progress(a + (b - a) * frac, stage)


def weights_path(kind, log=print, progress=no_progress):
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f"sam_{kind}.pth")
    if os.path.exists(dst) and os.path.getsize(dst) > 1e7:
        return dst
    last = None
    mb = 375 if kind == "vit_b" else 1250

    def hook(blocks, block_size, total):
        total = total if total > 0 else mb * 1e6
        progress(min(1.0, blocks * block_size / total),
                 f"downloading the segmentation model (once, {mb} MB)")
    for url in WEIGHTS[kind]:
        try:
            log(f"Downloading SAM {kind} weights from {url} (once, ~{mb} MB)...")
            urllib.request.urlretrieve(url, dst + ".part", reporthook=hook)
            os.replace(dst + ".part", dst)
            return dst
        except Exception as e:           # try the next source
            last = e
            log(f"  failed: {e}")
    raise RuntimeError(f"Could not download SAM weights: {last}. Put the file at {dst} by hand.")


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam(kind="vit_b", log=print, progress=no_progress):
    import torch
    from segment_anything import sam_model_registry
    if kind not in _models:
        path = weights_path(kind, log, progress)
        progress(1.0, "loading the segmentation model")
        sam = sam_model_registry[kind]()
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        sam.load_state_dict(sd)
        sam.eval()
        _models[kind] = sam
    return _models[kind]


def _run(fn, sam, log):
    """Run on the best device; fall back to CPU if the GPU path fails (MPS can)."""
    dev = device()
    try:
        sam.to(dev)
        return fn()
    except Exception as e:
        if dev == "cpu":
            raise
        log(f"  {dev} failed ({type(e).__name__}: {str(e)[:200]}), using the CPU instead")
        sam.to("cpu")
        return fn()


def _float32_points(gen):
    """segment_anything turns its grid of sample points into 64-bit floats before
    sending them to the device; Apple GPUs (MPS) have no 64-bit floats, which
    raises a TypeError. Make the point transform return 32-bit floats instead."""
    tr = gen.predictor.transform
    orig = tr.apply_coords
    tr.apply_coords = lambda coords, size: orig(coords, size).astype(np.float32)


# --------------------------------------------------------------------------- #
# Automatic segmentation
# --------------------------------------------------------------------------- #
def _report_batches(gen, progress):
    """SAM gives no progress of its own: wrap its two slow calls to report it.
    The image encoder runs once (0-30%), then the grid of sample points is
    processed in batches (30-95%), then masks are cleaned up."""
    n = math.ceil(gen.point_grids[0].shape[0] / gen.points_per_batch)
    done = [0]
    set_image, process_batch = gen.predictor.set_image, gen._process_batch

    def set_image_(*a, **k):
        done[0] = 0
        progress(0.02, "analysing the picture")
        out = set_image(*a, **k)
        progress(0.3, f"finding shapes (0 of {n})")
        return out

    def process_batch_(*a, **k):
        out = process_batch(*a, **k)
        done[0] += 1
        progress(0.3 + 0.65 * done[0] / n, f"finding shapes ({done[0]} of {n})")
        return out
    gen.predictor.set_image, gen._process_batch = set_image_, process_batch_


def auto_masks(image, kind="vit_b", points_per_side=32, log=print, progress=no_progress):
    import torch
    from segment_anything import SamAutomaticMaskGenerator
    sam = load_sam(kind, log, scaled(progress, 0, 0.05))
    img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    def go():
        gen = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side,
                                        pred_iou_thresh=0.82, stability_score_thresh=0.88,
                                        min_mask_region_area=int(img8.shape[0] * img8.shape[1] * 3e-4))
        _float32_points(gen)
        _report_batches(gen, scaled(progress, 0.05, 1))
        with torch.no_grad():
            masks = gen.generate(img8)
        progress(1.0, "cleaning up the shapes")
        return masks
    return _run(go, sam, log)


def partition(masks, image, min_frac):
    """Non-overlapping label map from overlapping SAM masks: big masks painted
    first, smaller (more detailed) ones on top. Areas no mask covers are split
    into colour-coherent pieces (Felzenszwalb superpixels) instead of becoming
    one big leftover region. Each connected piece becomes a region; pieces
    smaller than min_frac of the image join their neighbours. Regions are
    numbered 1..N from largest to smallest."""
    from skimage.segmentation import felzenszwalb
    shape = image.shape[:2]
    lab = np.zeros(shape, int)
    for k, m in enumerate(sorted(masks, key=lambda m: -m["area"]), 1):
        lab[m["segmentation"]] = k
    uncovered = lab == 0
    if uncovered.mean() > min_frac:
        scale = 0.6 * min_frac * lab.size          # bigger scale = bigger pieces
        sp = felzenszwalb(image, scale=scale, sigma=1.5, min_size=int(min_frac * lab.size / 2))
        lab = np.where(uncovered, lab.max() + 1 + sp, lab)
    return clean_labels(lab, min_frac)


def clean_labels(lab, min_frac):
    out = np.zeros(lab.shape, int)
    n = 0
    for k in np.unique(lab):
        cc, c = ndimage.label(lab == k)
        out[cc > 0] = cc[cc > 0] + n
        n += c
    min_px = min_frac * out.size
    for _ in range(6):
        sizes = np.bincount(out.ravel(), minlength=n + 1)
        small = sizes < min_px
        small[0] = True
        if not small[out].any():
            break
        keep = ~small[out]
        if not keep.any():
            break
        _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
        out = np.where(keep, out, out[iy, ix])
    ids, counts = np.unique(out, return_counts=True)
    remap = np.zeros(out.max() + 1, int)
    remap[ids[np.argsort(-counts)]] = np.arange(1, len(ids) + 1)
    return remap[out]


def adjacency(lab):
    """Set of (a, b) pairs of touching regions, a < b, with shared border length."""
    pairs = {}
    for A, B in ((lab[:-1], lab[1:]), (lab[:, :-1], lab[:, 1:])):
        diff = A != B
        a, b = np.minimum(A[diff], B[diff]), np.maximum(A[diff], B[diff])
        for p, c in zip(*np.unique(np.stack([a, b], 1), axis=0, return_counts=True)):
            pairs[tuple(p)] = pairs.get(tuple(p), 0) + c
    return pairs


def _border_share(lab):
    """For each region, the fraction of its own outline that lies on the picture
    border. Walls / sky / floor have a lot (> 0.25); an object that merely
    touches the edge (an arm reaching out of frame) has little."""
    n = lab.max() + 1
    ring = np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]
    on_border = np.bincount(ring, minlength=n)
    e = np.zeros(lab.shape, bool)
    d0 = lab[:-1] != lab[1:]
    d1 = lab[:, :-1] != lab[:, 1:]
    e[:-1] |= d0; e[1:] |= d0; e[:, :-1] |= d1; e[:, 1:] |= d1
    inner = np.bincount(lab[e], minlength=n)
    return on_border / np.maximum(on_border + inner, 1)


def _mean_lab(image, lab):
    from skimage.color import rgb2lab
    L = rgb2lab(image)
    n = lab.max() + 1
    cnt = np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
    return np.stack([np.bincount(lab.ravel(), L[..., k].ravel(), minlength=n) / cnt
                     for k in range(3)], -1)


def auto_groups(lab, image=None, background_min_pct=3.0, border_share=0.25, merge_de=18.0):
    """Default groups (the user can change everything afterwards):

    - background: regions with at least `border_share` of their own outline on
      the picture border, covering at least `background_min_pct` % of it (walls, sky, floor);
      one group each, except that backgrounds of similar colour (ΔE < merge_de) merge;
    - smaller regions that still run along the border (scraps of wall) join the
      background they touch with the closest colour, if it is close enough;
    - objects: everything else, grouped into clusters of touching regions.
    Objects come first in the order (largest first), backgrounds last."""
    n = lab.max()
    area = np.bincount(lab.ravel(), minlength=n + 1) / lab.size * 100
    share = _border_share(lab)
    col = _mean_lab(image, lab) if image is not None else np.zeros((n + 1, 3))
    adj = adjacency(lab)

    def de(a, b):
        return float(np.linalg.norm(col[a] - col[b]))
    bg = [r for r in range(1, n + 1) if share[r] >= border_share and area[r] >= background_min_pct]
    if not bg:
        bg = [int(np.argmax(share[1:]) + 1)]
    # merge similar backgrounds
    bg_sets = []
    for r in sorted(bg, key=lambda r: -area[r]):
        for S in bg_sets:
            if de(S[0], r) < merge_de and image is not None:
                S.append(r); break
        else:
            bg_sets.append([r])
    owner = {r: i for i, S in enumerate(bg_sets) for r in S}
    # border scraps
    for r in range(1, n + 1):
        if r in owner or share[r] < 0.8 * border_share:
            continue
        touching = [owner[b] for (x, y) in adj for b in (x, y)
                    if r in (x, y) and b != r and b in owner]
        if touching and image is not None:
            best = min(set(touching), key=lambda i: de(bg_sets[i][0], r))
            if de(bg_sets[best][0], r) < merge_de * 1.3:
                bg_sets[best].append(r); owner[r] = best
    rest = set(range(1, n + 1)) - set(owner)
    parent = {r: r for r in rest}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r
    for (a, b) in adj:
        if a in rest and b in rest:
            parent[find(a)] = find(b)
    clusters = {}
    for r in rest:
        clusters.setdefault(find(r), []).append(r)
    objects = sorted(clusters.values(), key=lambda ms: -sum(area[m] for m in ms))
    groups, order = [], 0
    for i, ms in enumerate(objects, 1):
        groups.append(new_group(len(groups) + 1, f"object {i}", "object", ms, order)); order += 1
    for i, S in enumerate(sorted(bg_sets, key=lambda S: -sum(area[m] for m in S)), 1):
        groups.append(new_group(len(groups) + 1, f"background {i}", "background", S, order)); order += 1
    return groups


def segment_project(project, log=print, reuse_masks=True, progress=no_progress):
    """Run SAM (or reuse its cached masks), build regions and default groups."""
    image = project.image()
    cache = project.path("sam_masks.npz")
    if reuse_masks and os.path.exists(cache):
        progress(0.1, "reusing the shapes found earlier")
        d = np.load(cache)
        masks = [{"segmentation": m, "area": int(a)} for m, a in zip(d["masks"], d["areas"])]
        log(f"reusing {len(masks)} cached SAM masks")
    else:
        masks = auto_masks(image, project.cfg["sam_model"], project.cfg["sam_points_per_side"], log,
                           scaled(progress, 0, 0.85))
        np.savez_compressed(cache, masks=np.array([m["segmentation"] for m in masks]),
                            areas=np.array([m["area"] for m in masks]))
        log(f"SAM found {len(masks)} masks")
    progress(0.88, "turning shapes into numbered regions")
    lab = partition(masks, image, project.cfg["min_region_pct"] / 100)
    project.set_regions(lab)
    progress(0.95, "grouping regions into objects and backgrounds")
    project.groups = auto_groups(lab, image, project.cfg["background_min_pct"])
    project.save()
    progress(1.0, "done")
    log(f"{lab.max()} regions in {len(project.groups)} groups")


# --------------------------------------------------------------------------- #
# Click refinement: SAM mask from one point
# --------------------------------------------------------------------------- #
def mask_from_click(project, x, y, log=print, progress=no_progress):
    """SAM mask for the object under pixel (x, y) of the working image."""
    import torch
    from segment_anything import SamPredictor
    kind = project.cfg["sam_model"]
    sam = load_sam(kind, log, scaled(progress, 0, 0.1))
    key = (project.path("image.png"), os.path.getmtime(project.path("image.png")), kind)
    if key not in _predictor_cache:
        _predictor_cache.clear()
        progress(0.1, "analysing the picture (first click only, can take a minute)")
        pred = SamPredictor(sam)
        img8 = (project.image() * 255).astype(np.uint8)
        _run(lambda: pred.set_image(img8), sam, log)
        _predictor_cache[key] = pred
    pred = _predictor_cache[key]
    progress(0.9, "finding the shape under the click")
    with torch.no_grad():
        masks, scores, _ = pred.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]),
                                        multimask_output=True)
    return masks[int(np.argmax(scores))]


def add_region_from_mask(project, mask):
    """Carve a new region out of the label map; returns its id. It joins the
    group that owned most of those pixels (the user can then move it)."""
    lab = project.regions()
    new = lab.max() + 1
    before = lab[mask]
    owner = None
    if before.size:
        vals, cnt = np.unique(before, return_counts=True)
        owner = project.group_of(int(vals[np.argmax(cnt)]))
    lab = np.where(mask, new, lab)
    if owner is not None:
        owner["members"].append(int(new))
    G.drop_missing(project, lab)      # regions that lost all their pixels leave their groups
    project.set_regions(lab)
    project.save()
    return int(new)
