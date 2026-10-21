"""Direction field for tile alignment, built from a neural edge map (e.g. SAUGE).

Pipeline (each step is a separate function so it can be inspected/plotted):

    1. normalize      edge map rescaled to [0, 1] (1 = edge), image border cleared
    2. threshold      pixels below `threshold` (default 30%) are dropped
    3. skeletonize    holes smaller than `hole_area` tiles² are filled (they would
                      become little loops), then the bands are thinned to
                      1-pixel centerlines
    4. prune          side branches ("spurs") shorter than `spur_length` tiles are
                      cut, then pieces shorter than `min_length` tiles are removed
    5. direction      at each line pixel, the line's local direction, from a
                      structure tensor smoothed over `smoothing` tiles
    6. echo           every pixel takes the direction of its NEAREST line pixel,
                      with weight exp(-distance / echo)  (distance and echo in
                      tile widths). Small echo -> only tiles on the lines are
                      constrained; large echo -> rows of tiles echo the contours
                      outward (opus vermiculatum).

The result is an (H, W, 2) field of vectors in mosaic directions (x right, y UP):
direction = line direction, length = weight in [0, 1]. Pass it to
`loss.alignment_loss(..., weighting="magnitude")` (or total_loss with
field=..., softness=None, weighting="magnitude").

This is pre-processing: computed once per image with numpy/scipy/scikit-image,
not differentiated through.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def normalize(edge: np.ndarray, border_px: int = 10) -> np.ndarray:
    """Rescale to [0, 1] and zero a border strip (SAUGE leaves dotted artifacts there)."""
    e = (edge - edge.min()) / (edge.max() - edge.min() + 1e-12)
    if border_px > 0:
        e = e.copy()
        e[:border_px, :] = e[-border_px:, :] = 0
        e[:, :border_px] = e[:, -border_px:] = 0
    return e


def threshold(edge: np.ndarray, t: float = 0.3) -> np.ndarray:
    """Boolean mask of pixels at least `t` of the way from 'no edge' to 'edge'."""
    return edge >= t


_EIGHT = np.ones((3, 3), dtype=bool)


def fill_small_holes(mask: np.ndarray, max_area_px: float) -> np.ndarray:
    """Fill background holes (not touching the image border) up to max_area_px pixels."""
    labels, _ = ndimage.label(~mask)
    sizes = np.bincount(labels.ravel())
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    small = sizes <= max_area_px
    small[0] = False
    small[border] = False
    return mask | small[labels]


def skeleton(mask: np.ndarray, hole_area_px: float = 0) -> np.ndarray:
    """Fill small holes, then thin the mask to 1-pixel-wide centerlines."""
    if hole_area_px > 0:
        mask = fill_small_holes(mask, hole_area_px)
    return skeletonize(mask)


def _neighbours(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours of each skeleton pixel."""
    return ndimage.convolve(skel.astype(int), _EIGHT.astype(int), mode="constant") - skel


def remove_spurs(skel: np.ndarray, max_length_px: float, rounds: int = 3) -> np.ndarray:
    """Cut short side branches: a branch that ends in a free endpoint and is at most
    max_length_px long is removed, as long as it hangs off a junction (so short
    isolated lines are left to `prune`). Repeated a few times, since removing a
    spur can turn its junction into a plain line point."""
    skel = skel.copy()
    for _ in range(rounds):
        nb = _neighbours(skel)
        junction = skel & (nb >= 3)
        endpoint = skel & (nb == 1)
        # split the skeleton into branches at (slightly thickened) junctions
        jz = ndimage.binary_dilation(junction, _EIGHT) & skel
        branches, n = ndimage.label(skel & ~jz, structure=_EIGHT)
        if n == 0:
            break
        sizes = np.bincount(branches.ravel())
        has_end = np.zeros(n + 1, bool)
        has_end[np.unique(branches[endpoint])] = True
        touches_junction = np.zeros(n + 1, bool)
        touches_junction[np.unique(branches[ndimage.binary_dilation(jz, _EIGHT)])] = True
        spur = has_end & touches_junction & (sizes <= max_length_px)
        spur[0] = False
        if not spur.any():
            break
        skel &= ~spur[branches]
        skel = skeletonize(skel)        # tidy the junction pixels left behind
    return skel


def prune(skel: np.ndarray, min_length_px: float) -> np.ndarray:
    """Remove connected skeleton pieces with fewer than `min_length_px` pixels."""
    labels, n = ndimage.label(skel, structure=np.ones((3, 3)))   # 8-connected pieces
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_length_px
    keep[0] = False                                                # background
    return keep[labels]


def line_direction(skel: np.ndarray, smoothing_px: float):
    """Local line direction at every pixel, as a doubled angle.

    Structure tensor J = G_s * (∇s ∇sᵀ) of the (slightly blurred) skeleton.
    Returns (c2, s2, coherence):
        (c2, s2) = (cos 2ψ, sin 2ψ) where ψ is the direction ACROSS the line
                   (image coordinates, y down)
        coherence in [0, 1]: 1 = clear single direction, 0 = ambiguous (junctions)
    Doubled angles are used because a line has no 'forward' direction.
    """
    s = ndimage.gaussian_filter(skel.astype(float), 1.0)
    gy, gx = np.gradient(s)
    jxx = ndimage.gaussian_filter(gx * gx, smoothing_px)
    jyy = ndimage.gaussian_filter(gy * gy, smoothing_px)
    jxy = ndimage.gaussian_filter(gx * gy, smoothing_px)
    a, b = jxx - jyy, 2 * jxy
    r = np.sqrt(a * a + b * b)
    c2, s2 = a / (r + 1e-12), b / (r + 1e-12)
    coherence = r / (jxx + jyy + 1e-12)
    return c2, s2, coherence


def echo(skel: np.ndarray, c2: np.ndarray, s2: np.ndarray,
         px_per_tile: float, echo_tiles: float):
    """Propagate the line direction to every pixel from its nearest line pixel.

    Returns (c2_all, s2_all, distance_tiles, weight) with
    weight = exp(-distance / echo_tiles)  (echo_tiles = inf -> weight 1 everywhere).
    """
    dist_px, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    d = dist_px / px_per_tile
    weight = np.ones_like(d) if np.isinf(echo_tiles) else np.exp(-d / echo_tiles)
    return c2[iy, ix], s2[iy, ix], d, weight


def to_field(c2: np.ndarray, s2: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Doubled-angle (image coords) -> (H, W, 2) vectors in mosaic coords (y up).

    Any vector along the line or across it describes the same square alignment,
    so we simply use the 'across' direction ψ = atan2(s2, c2) / 2."""
    psi = 0.5 * np.arctan2(s2, c2)
    vx, vy = np.cos(psi), -np.sin(psi)          # flip y: image rows point down
    return np.stack([vx, vy], -1) * weight[..., None]


def build_edge_field(edge_map: np.ndarray, board_width: float, tile_width: float,
                     threshold_value: float = 0.3, hole_area: float = 1.0,
                     spur_length: float = 0.75, min_length: float = 1.0,
                     smoothing: float = 0.5, echo_tiles: float = 1.0,
                     border_px: int = 10, return_steps: bool = False):
    """Full pipeline. Lengths (spur_length, min_length, smoothing, echo_tiles) are in
    tile widths, hole_area in tile widths squared.

    edge_map: (H, W) array, larger = more edge (e.g. SAUGE output), stretched
              over the whole board like the image.
    """
    h, w = edge_map.shape
    px_per_tile = w / board_width * tile_width
    e = normalize(edge_map, border_px)
    mask = threshold(e, threshold_value)
    sk = skeleton(mask, hole_area * px_per_tile ** 2)
    sk_p = prune(remove_spurs(sk, spur_length * px_per_tile), min_length * px_per_tile)
    c2, s2, coh = line_direction(sk_p, smoothing * px_per_tile)
    c2a, s2a, dist, weight = echo(sk_p, c2, s2, px_per_tile, echo_tiles)
    field = to_field(c2a, s2a, weight)
    if not return_steps:
        return field
    return field, dict(normalized=e, mask=mask, skeleton=sk, pruned=sk_p,
                       c2=c2, s2=s2, coherence=coh, distance=dist, weight=weight,
                       px_per_tile=px_per_tile)
