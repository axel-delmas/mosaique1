"""Place tiles on a grid that bends to follow a direction field.

2D version of the position-field step of "Instant Field-Aligned Meshes"
(Jakob, Tarini, Panozzo, Sorkine-Hornung, 2015).

The board is covered with sample points (a few per tile). Each sample i has
    x_i : its position
    θ_i : the local orientation (from the edge field, smoothed; mod 90°)
    q_i : the origin of a LOCAL square lattice with spacing s, rotated by θ_i
Neighbouring samples repeatedly nudge their lattices to agree (up to whole
lattice steps and 90° turns, which don't change a square lattice). Once they
agree, the lattice points near the samples are where tiles go. The number of
tiles is not fixed in advance: it comes from how the lattice bends.

Pipeline:
    orientation_on_samples  edge field -> smoothed 4-way orientation per sample
    solve_position_field    lattice origins q_i, made consistent by smoothing
    extract_tiles           lattice points -> clustered tile centers + angles
    deformed_grid_mosaic    all of the above -> a Mosaic, ready for polishing
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from loss import sample_image, tile_sample_points
from mosaic import Mosaic, Tile


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
def sample_grid(board_w: float, board_h: float, spacing: float):
    """Regular sample points (ny, nx, 2) covering the board, row 0 at the bottom."""
    xs = np.arange(spacing / 2, board_w, spacing)
    ys = np.arange(spacing / 2, board_h, spacing)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], -1)


def orientation_on_samples(field: np.ndarray, X: np.ndarray, board_w: float,
                           board_h: float, smooth_samples: float = 2.0) -> np.ndarray:
    """Angle θ (radians, mosaic coords, meaningful mod 90°) at each sample.

    The field is sampled at the points, turned into the 4-way representation
    e^{4iθ} (so θ and θ + 90° are the same), blurred over `smooth_samples`
    samples to remove seams, and turned back into an angle.
    """
    f = sample_image(torch.tensor(field, dtype=torch.float32),
                     torch.tensor(X, dtype=torch.float32), board_w, board_h).numpy()
    ang = np.arctan2(f[..., 1], f[..., 0])
    wgt = np.linalg.norm(f, axis=-1)
    c = ndimage.gaussian_filter(wgt * np.cos(4 * ang), smooth_samples, mode="nearest")
    s = ndimage.gaussian_filter(wgt * np.sin(4 * ang), smooth_samples, mode="nearest")
    return np.arctan2(s, c) / 4


# --------------------------------------------------------------------------- #
# Position field
# --------------------------------------------------------------------------- #
def _round_to_lattice(q, theta, p, s):
    """Lattice point (origin q, axes rotated by theta, spacing s) closest to p.
    q, p: (..., 2); theta: (...)."""
    c, sn = np.cos(theta), np.sin(theta)
    d = p - q
    a = np.round((d[..., 0] * c + d[..., 1] * sn) / s)
    b = np.round((-d[..., 0] * sn + d[..., 1] * c) / s)
    return q + s * np.stack([a * c - b * sn, a * sn + b * c], -1)


def _shift(arr, dy, dx):
    """arr shifted so that out[y, x] = arr[y + dy, x + dx]; also returns validity mask."""
    ny, nx = arr.shape[:2]
    out = np.zeros_like(arr)
    valid = np.zeros((ny, nx), bool)
    ys, yd = (slice(dy, ny), slice(0, ny - dy)) if dy >= 0 else (slice(0, ny + dy), slice(-dy, ny))
    xs, xd = (slice(dx, nx), slice(0, nx - dx)) if dx >= 0 else (slice(0, nx + dx), slice(-dx, nx))
    out[yd, xd] = arr[ys, xs]
    valid[yd, xd] = True
    return out, valid


def propagate_init(X: np.ndarray, theta: np.ndarray, spacing: float) -> np.ndarray:
    """Starting lattice origins, grown outward from the board center (breadth-first).

    Each newly reached sample j takes over the lattice of the neighbour i it was
    reached from (i's lattice point nearest x_j, then re-oriented with θ_j), so
    neighbours start out consistent. Leftover mismatches are concentrated where the
    field turns a lot, and the smoothing then relaxes them.
    """
    from collections import deque
    ny, nx = theta.shape
    Q = np.zeros_like(X)
    seen = np.zeros((ny, nx), bool)
    start = (ny // 2, nx // 2)
    Q[start] = X[start]
    seen[start] = True
    queue = deque([start])
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y, x = queue.popleft()
        for dy, dx in nbrs:
            v, u = y + dy, x + dx
            if 0 <= v < ny and 0 <= u < nx and not seen[v, u]:
                q = _round_to_lattice(Q[y, x], theta[y, x], X[v, u], spacing)
                Q[v, u] = q
                seen[v, u] = True
                queue.append((v, u))
    return Q


def solve_position_field(X: np.ndarray, theta: np.ndarray, spacing: float,
                         iterations: int = 50, init: str = "propagate",
                         inlier: float = 0.2, seed: int = 0) -> np.ndarray:
    """Lattice origins q (ny, nx, 2), smoothed so neighbouring lattices agree.

    Start: propagate_init (default) or random. Then red-black sweeps over the 8
    neighbours. For sample i and neighbour j, q_j (a point of j's lattice) is
    compared with the nearest point of i's lattice; the difference is how far i's
    lattice would have to move to agree with j. (Comparing at lattice points, not
    at the midpoint between samples, avoids ambiguous rounding when the midpoint
    falls exactly halfway between two lattice points.) These 9 candidate lattices (i's
    own + 8 neighbours') are then VOTED on, not averaged: i takes the candidate
    that most others agree with (within `inlier` x spacing) and averages only that
    group. Plain averaging would smear every seam into a slow drift of the
    lattice; voting keeps seams sharp and lets the lattice relax inside
    consistent regions. Finally q_i is re-centered on its lattice point nearest x_i.
    """
    if init == "propagate":
        Q = propagate_init(X, theta, spacing)
    else:                                                       # random start, as in the paper
        rng = np.random.default_rng(seed)
        Q = X + rng.uniform(-0.5, 0.5, X.shape) * spacing
    ny, nx = theta.shape
    red = (np.add.outer(np.arange(ny), np.arange(nx)) % 2) == 0
    nbrs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    tol = inlier * spacing
    for _ in range(iterations):
        for colour in (red, ~red):
            cands = [Q]
            valids = [np.ones((ny, nx), bool)]
            for dy, dx in nbrs:
                Qj, valid = _shift(Q, dy, dx)
                qi = _round_to_lattice(Q, theta, Qj, spacing)
                cands.append(Q + (Qj - qi))
                valids.append(valid)
            C = np.stack(cands, 2)                              # (ny, nx, 9, 2)
            V = np.stack(valids, 2)                             # (ny, nx, 9)
            d = np.linalg.norm(C[:, :, :, None] - C[:, :, None], axis=-1)
            agree = (d < tol) & V[:, :, :, None] & V[:, :, None, :]
            votes = agree.sum(-1) + np.where(V, 0, -99)
            best = votes.argmax(-1)                             # (ny, nx)
            inl = np.take_along_axis(agree, best[..., None, None], 2)[:, :, 0]   # (ny, nx, 9)
            mean = (C * inl[..., None]).sum(2) / inl.sum(-1, keepdims=True)
            Qn = _round_to_lattice(mean, theta, X, spacing)
            Q = np.where(colour[..., None], Qn, Q)
    return Q


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #
def seam_mask(Q: np.ndarray, theta: np.ndarray, spacing: float,
              tolerance: float = 0.25) -> np.ndarray:
    """Samples whose lattice disagrees with a right/upper neighbour's by more than
    tolerance x spacing (compared at the neighbour's lattice point)."""
    seam = np.zeros(theta.shape, bool)
    for dy, dx in [(0, 1), (1, 0)]:
        Qj, v = _shift(Q, dy, dx)
        d = np.linalg.norm(Qj - _round_to_lattice(Q, theta, Qj, spacing), axis=-1)
        seam |= v & (d > tolerance * spacing)
    return seam


def extract_tiles(Q: np.ndarray, theta: np.ndarray, spacing: float,
                  board_w: float, board_h: float, merge: float = 0.35,
                  min_support: int = 2, min_gap: float = 0.75):
    """Lattice points -> tile centers and angles.

    Every sample votes for its lattice point nearest to itself (= q_i). Votes
    closer than `merge`·spacing are clustered; a cluster with at least
    `min_support` votes becomes a tile (center = mean of the votes, angle =
    4-way mean of the voters' angles). Finally tiles closer than
    `min_gap`·spacing are thinned (the better supported one is kept) and tiles
    whose center is off the board are dropped.
    """
    pts = Q.reshape(-1, 2)
    th = theta.reshape(-1)
    pairs = cKDTree(pts).query_pairs(merge * spacing, output_type="ndarray")
    n = len(pts)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    support = np.bincount(lab)
    centers = np.stack([np.bincount(lab, pts[:, k]) for k in (0, 1)], -1) / support[:, None]
    c4 = np.bincount(lab, np.cos(4 * th))
    s4 = np.bincount(lab, np.sin(4 * th))
    angles = np.arctan2(s4, c4) / 4
    keep = support >= min_support
    centers, angles, support = centers[keep], angles[keep], support[keep]

    order = np.argsort(-support)                     # thin: best supported first
    close = cKDTree(centers).query_ball_point(centers, min_gap * spacing)
    removed = np.zeros(len(centers), bool)
    chosen = []
    for k in order:
        if removed[k]:
            continue
        chosen.append(k)
        removed[close[k]] = True
    centers, angles = centers[chosen], angles[chosen]
    inside = ((centers[:, 0] > 0) & (centers[:, 0] < board_w) &
              (centers[:, 1] > 0) & (centers[:, 1] < board_h))
    return centers[inside], angles[inside]


def mean_colors(centers, angles, image: torch.Tensor, tile_width, board_w, board_h):
    """Average image color under each tile (the best flat color for the L2 loss)."""
    pts = tile_sample_points(torch.tensor(centers, dtype=torch.float32),
                             torch.tensor(angles, dtype=torch.float32), tile_width)
    return sample_image(image, pts, board_w, board_h).mean(1).numpy()


def deformed_grid_mosaic(field: np.ndarray, image: torch.Tensor, board_w: float,
                         board_h: float, tile_width: float, spacing: float,
                         samples_per_tile: int = 3, iterations: int = 50,
                         smooth_samples: float = 2.0, unit: str = "cm",
                         return_steps: bool = False):
    """Full pipeline: direction field -> Mosaic on a field-aligned deformed grid."""
    X = sample_grid(board_w, board_h, spacing / samples_per_tile)
    theta = orientation_on_samples(field, X, board_w, board_h, smooth_samples)
    Q = solve_position_field(X, theta, spacing, iterations)
    centers, angles = extract_tiles(Q, theta, spacing, board_w, board_h)
    cols = mean_colors(centers, angles, image, tile_width, board_w, board_h)
    tiles = [Tile(color=tuple(int(round(v * 255)) for v in np.clip(c, 0, 1)),
                  position=tuple(p), orientation=math.degrees(a))
             for p, a, c in zip(centers, angles, cols)]
    m = Mosaic(board_w, board_h, tile_width, tiles, unit)
    if return_steps:
        return m, dict(X=X, theta=theta, Q=Q, centers=centers, angles=angles)
    return m
