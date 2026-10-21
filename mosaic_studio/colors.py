"""Step 5: tile colours and exports.

  * edge-preserving smoothing of the picture (removes brush/photo texture)
  * each tile takes the median colour under it, saturation slightly boosted
  * each group gets its own limited palette (k-means in Lab, `colors` per group)
  * outline contrast: the first row of a group, where it borders another group,
    is pushed lighter or darker (whichever increases contrast with the colour
    across the border), choosing among its own palette colours
  * exports: mosaic.png, tiles.csv, shopping_list.csv
"""
from __future__ import annotations

import csv

import numpy as np
from scipy import ndimage
from skimage.color import lab2rgb, rgb2lab

from .geometry import _lookup


def smooth_image(image, passes=3, diameter=9, sigma_color=25, sigma_space=6):
    img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    try:
        import cv2
        for _ in range(passes):
            img = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
        return img.astype(np.float32) / 255
    except ImportError:
        from skimage.restoration import denoise_bilateral
        out = img.astype(np.float32) / 255
        for _ in range(passes):
            out = denoise_bilateral(out, win_size=diameter, sigma_color=sigma_color / 255,
                                    sigma_spatial=sigma_space, channel_axis=-1)
        return out.astype(np.float32)


def sample_colors(image, pts_cm, ppc, board_h):
    """Nearest-pixel colours at board points (..., 2) -> (..., 3)."""
    H, W = image.shape[:2]
    shp = pts_cm.shape[:-1]
    p = pts_cm.reshape(-1, 2)
    r = np.clip(((board_h - p[:, 1]) * ppc - 0.5).round().astype(int), 0, H - 1)
    c = np.clip((p[:, 0] * ppc - 0.5).round().astype(int), 0, W - 1)
    return image[r, c].reshape(*shp, 3)


def tile_sample_points(tiles, per_side=4):
    k = per_side
    t = (np.arange(k) + 0.5) / k - 0.5
    gx, gy = np.meshgrid(t, t)
    local = np.stack([gx.ravel(), gy.ravel()], -1)                   # (S, 2) in tile units
    X = np.array([[t_["x"], t_["y"]] for t_ in tiles])
    A = np.array([t_["angle"] for t_ in tiles])
    S = np.array([t_["size"] for t_ in tiles])
    c, s = np.cos(A), np.sin(A)
    lx, ly = local[None, :, 0] * S[:, None], local[None, :, 1] * S[:, None]
    return np.stack([X[:, None, 0] + lx * c[:, None] - ly * s[:, None],
                     X[:, None, 1] + lx * s[:, None] + ly * c[:, None]], -1)


def kmeans(lab, k, iters=25, seed=0):
    k = max(1, min(k, len(lab)))
    rng = np.random.default_rng(seed)
    C = [lab[rng.integers(len(lab))]]
    for _ in range(1, k):
        d = np.min(((lab[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        if d.sum() == 0:
            break
        C.append(lab[rng.choice(len(lab), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        idx = ((lab[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(len(C)):
            if (idx == j).any():
                C[j] = lab[idx == j].mean(0)
    return C[np.argsort(C[:, 0])]


def color_tiles(project, tiles, log=print):
    """Adds 'color' (r, g, b 0-255) and 'palette' (group, index) to every tile."""
    cfg = project.cfg
    groups = project.ordered_groups()
    board_w, board_h, ppc = project.board()
    image = project.image()
    img = smooth_image(image) if cfg["smooth_image"] else image
    pts = tile_sample_points(tiles)
    raw = np.median(sample_colors(img, pts, ppc, board_h), 1)
    lab = rgb2lab(np.clip(raw, 0, 1)[None])[0]
    lab[:, 1:] *= float(cfg["saturation"])
    gid = np.array([t["group"] for t in tiles])
    row = np.array([t["row"] for t in tiles])

    # colour across the nearest border with another group, for outline tiles
    gmap = project.group_map()
    border = np.zeros(gmap.shape, bool)
    d0 = gmap[:-1] != gmap[1:]
    d1 = gmap[:, :-1] != gmap[:, 1:]
    border[:-1] |= d0; border[1:] |= d0; border[:, :-1] |= d1; border[:, 1:] |= d1
    dist_px, (iy, ix) = ndimage.distance_transform_edt(~border, return_indices=True)
    X = np.array([[t["x"], t["y"]] for t in tiles])
    H, W = gmap.shape
    r = np.clip(((board_h - X[:, 1]) * ppc - 0.5).astype(int), 0, H - 1)
    c = np.clip((X[:, 0] * ppc - 0.5).astype(int), 0, W - 1)
    near_border = dist_px[r, c] / ppc < 1.2 * float(cfg["tile_cm"])
    P = np.stack([(ix[r, c] + 0.5) / ppc, board_h - (iy[r, c] + 0.5) / ppc], -1)
    across = rgb2lab(np.clip(sample_colors(img, 2 * P - X, ppc, board_h), 0, 1)[None])[0]

    out_rgb = np.zeros((len(tiles), 3))
    pal_idx = np.zeros(len(tiles), int)
    palettes = []
    for gi, g in enumerate(groups):
        sel = np.flatnonzero(gid == gi)
        if not len(sel):
            palettes.append(np.zeros((0, 3)))
            continue
        pal = kmeans(lab[sel], int(g["colors"]), seed=gi)
        idx = ((lab[sel, None] - pal[None]) ** 2).sum(-1).argmin(1)
        # outline contrast
        strength = float(cfg["contrast"])
        if strength > 0:
            for n, i in enumerate(sel):
                if row[i] != 0 or not near_border[i]:
                    continue
                direction = 1.0 if lab[i, 0] >= across[i, 0] else -1.0
                target = lab[i].copy(); target[0] += direction * strength
                ok = direction * (pal[:, 0] - lab[i, 0]) >= strength / 2
                if ok.any():
                    cand = np.flatnonzero(ok)
                    idx[n] = cand[((pal[cand] - target) ** 2).sum(-1).argmin()]
        rgb = np.clip(lab2rgb(pal[None])[0], 0, 1)
        out_rgb[sel] = rgb[idx]
        pal_idx[sel] = idx
        palettes.append(rgb)
    for t, col, pi in zip(tiles, out_rgb, pal_idx):
        t["color"] = tuple(int(round(v * 255)) for v in col)
        t["palette"] = int(pi)
    log(f"  colours: {sum(len(p) for p in palettes)} palette colours over {len(groups)} groups")
    return tiles


def hexcol(c):
    return "#%02x%02x%02x" % tuple(c)


def export(project, tiles):
    """tiles.csv and shopping_list.csv in the project folder. Coordinates in cm,
    origin at the bottom-left corner of the board, angles in degrees."""
    groups = project.ordered_groups()
    with open(project.path("tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_cm", "y_cm", "angle_deg", "size_cm", "color_hex", "r", "g", "b", "group"])
        for i, t in enumerate(tiles, 1):
            w.writerow([i, f"{t['x']:.3f}", f"{t['y']:.3f}", f"{np.degrees(t['angle']) % 90:.1f}",
                        f"{t['size']:.3f}", hexcol(t["color"]), *t["color"], groups[t["group"]]["name"]])
    counts = {}
    for t in tiles:
        k = hexcol(t["color"])
        e = counts.setdefault(k, {"n": 0, "groups": set(), "rgb": t["color"]})
        e["n"] += 1
        e["groups"].add(groups[t["group"]]["name"])
    with open(project.path("shopping_list.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["color_hex", "r", "g", "b", "tiles", "used_in"])
        for k, e in sorted(counts.items(), key=lambda kv: -kv[1]["n"]):
            w.writerow([k, *e["rgb"], e["n"], "; ".join(sorted(e["groups"]))])
    return counts
