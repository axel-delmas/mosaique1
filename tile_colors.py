"""Tile colors that make objects read: smoothing, median, saturation, palette,
and outlines that contrast in the right direction.

    smooth_image      edge-preserving smoothing (repeated bilateral filter, the
                      classic "cartoon" filter): removes brush texture inside
                      regions, keeps the boundaries between them sharp
    tile_raw_colors   median color under each tile (not the mean: less muddy)
    boost_saturation  multiply chroma in Lab (averaging washes colors out)
    kmeans_palette    a limited set of tile colors, like a mosaicist's stock
    quantize          every tile -> nearest palette color
    outline_contrast  tiles in the row next to a contour look at the color just
                      across that contour. If they are darker than the other side
                      they take a DARKER palette color of similar hue, if lighter a
                      LIGHTER one. So a light dancer on a dark wall gets a bright
                      edge and the wall a dark one, and the other way round —
                      never a fixed "dark outline".
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from skimage.color import lab2rgb, rgb2lab

from loss import sample_image, tile_sample_points


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def smooth_image(image: np.ndarray, passes: int = 3, diameter: int = 9,
                 sigma_color: float = 25, sigma_space: float = 6) -> np.ndarray:
    """Edge-preserving smoothing of an (H, W, 3) float image in [0, 1].

    Uses OpenCV's bilateral filter when available (fast), otherwise
    scikit-image's (slower). sigma_color is in 0-255 units."""
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


# --------------------------------------------------------------------------- #
# Per-tile colors
# --------------------------------------------------------------------------- #
def _samples(image, centers, angles, tile_w, board_w, board_h, per_side=5):
    pts = tile_sample_points(torch.tensor(centers, dtype=torch.float32),
                             torch.tensor(angles, dtype=torch.float32), tile_w, per_side)
    return sample_image(torch.tensor(image), pts, board_w, board_h).numpy()   # (N, S, 3)


def tile_raw_colors(image, centers, angles, tile_w, board_w, board_h, stat="median"):
    """Median (or mean) RGB of the image under each tile."""
    s = _samples(image, centers, angles, tile_w, board_w, board_h)
    return np.median(s, 1) if stat == "median" else s.mean(1)


def boost_saturation(rgb: np.ndarray, factor: float = 1.2) -> np.ndarray:
    lab = rgb2lab(rgb[None])[0]
    lab[:, 1:] *= factor
    return np.clip(lab2rgb(lab[None])[0], 0, 1)


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
def kmeans_palette(lab: np.ndarray, k: int = 16, iters: int = 30, seed: int = 0) -> np.ndarray:
    """k-means (k-means++ start) on Lab colors -> (k, 3) palette in Lab."""
    rng = np.random.default_rng(seed)
    C = [lab[rng.integers(len(lab))]]
    for _ in range(1, k):
        d = np.min(((lab[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(lab[rng.choice(len(lab), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab_idx = ((lab[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab_idx == j).any():
                C[j] = lab[lab_idx == j].mean(0)
    order = np.argsort(C[:, 0])                   # dark to light, nicer swatches
    return C[order]


def quantize(lab: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color for each Lab color."""
    return ((lab[:, None] - palette[None]) ** 2).sum(-1).argmin(1)


# --------------------------------------------------------------------------- #
# Outline contrast
# --------------------------------------------------------------------------- #
def other_side_colors(image, centers, angles, tile_w, skeleton, ppc, board_w, board_h,
                      border_px=2):
    """For each tile, the image color mirrored across its nearest contour point
    (as far beyond the contour as the tile is before it). Also returns a mask of
    tiles whose nearest contour is the board edge (no 'other side' there)."""
    _, (iy, ix) = ndimage.distance_transform_edt(~skeleton, return_indices=True)
    H, W = skeleton.shape
    r = np.clip(((board_h - centers[:, 1]) * ppc - 0.5).astype(int), 0, H - 1)
    c = np.clip((centers[:, 0] * ppc - 0.5).astype(int), 0, W - 1)
    py, px = iy[r, c], ix[r, c]
    on_frame = (py < border_px) | (py >= H - border_px) | (px < border_px) | (px >= W - border_px)
    p = np.stack([(px + 0.5) / ppc, board_h - (py + 0.5) / ppc], -1)
    mirrored = 2 * p - centers
    return tile_raw_colors(image, mirrored, angles, tile_w, board_w, board_h), on_frame


def outline_contrast(idx, lab, other_lab, palette, apply, strength=12.0, min_step=6.0):
    """Push outline tiles away from the other side of their contour, in lightness,
    choosing among palette colors only.

    For each tile in `apply`: direction = sign(L_tile - L_other). The target is the
    tile's color with L moved `strength` further in that direction; the chosen
    palette color is the nearest to the target among those at least `min_step`
    lighter (or darker) than the tile. If no palette color qualifies, the tile's
    usual palette color is kept."""
    out = idx.copy()
    for i in np.flatnonzero(apply):
        direction = 1.0 if lab[i, 0] >= other_lab[i, 0] else -1.0
        target = lab[i].copy()
        target[0] += direction * strength
        ok = direction * (palette[:, 0] - lab[i, 0]) >= min_step
        if ok.any():
            cand = np.flatnonzero(ok)
            out[i] = cand[((palette[cand] - target) ** 2).sum(-1).argmin()]
    return out


# --------------------------------------------------------------------------- #
# Everything
# --------------------------------------------------------------------------- #
def color_tiles(image, centers, angles, rows, tile_w, board_w, board_h, skeleton, ppc,
                k=16, saturation=1.2, outline_rows=(0,), strength=12.0, smooth=True,
                return_steps=False):
    """Final RGB per tile (N, 3) in [0, 1]."""
    img = smooth_image(image) if smooth else image.astype(np.float32)
    raw = tile_raw_colors(img, centers, angles, tile_w, board_w, board_h)
    rgb = boost_saturation(raw, saturation)
    lab = rgb2lab(rgb[None])[0]
    palette = kmeans_palette(lab, k)
    idx = quantize(lab, palette)
    other, on_frame = other_side_colors(img, centers, angles, tile_w, skeleton, ppc,
                                        board_w, board_h)
    other_lab = rgb2lab(boost_saturation(other, saturation)[None])[0]
    apply = np.isin(rows, outline_rows) & ~on_frame
    idx_out = outline_contrast(idx, lab, other_lab, palette, apply, strength)
    pal_rgb = np.clip(lab2rgb(palette[None])[0], 0, 1)
    final = pal_rgb[idx_out]
    if not return_steps:
        return final
    return final, dict(smoothed=img, raw=raw, boosted=rgb, palette_rgb=pal_rgb,
                       quantized=pal_rgb[idx], outline_mask=apply,
                       changed=(idx_out != idx))
