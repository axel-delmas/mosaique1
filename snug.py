"""Greedy "snugging" of placed tiles, alternated with gap filling.

relax_sweep: visit the tiles one by one (greedy). For each tile try a small set
of moves: shifts of 0 / ±step along each of its own two axes (9 positions)
times rotations of 0 / ±a1 / ±a2 (5 angles) = 45 candidate poses. A pose is
allowed only if the tile does not overlap any neighbour (same shrunk-square
test as the placement). Among allowed poses the one with the lowest score is
kept (the current pose is always a candidate, so a tile only moves if that
strictly helps):

    score = gap_weight  * mean over close neighbours of ((gap - grout) / w)²
          + align_weight * mean over close neighbours of (1 - cos 4(θ - θ_j)) / 2

gap = clearance between the two squares (largest separation over the 4
separating axes; exact for side-by-side tiles). So a tile is pulled towards
neighbours that are too far, pushed from ones that are too close, and turned
to match its peers.

snug_and_fill: sweep, fill the gaps that opened up, sweep again ... until a
round moves almost nothing and adds no tile.
"""
from __future__ import annotations

import math

import numpy as np

from guide_lines import fill_gaps


def _pair_clearance(xy, ang, NXY, NA, half, shrink=1.0):
    """Clearance between K candidate squares (xy (K,2), ang (K,)) and M neighbours
    (NXY (M,2), NA (M,)) -> (K, M). Negative = overlap (for the shrunk squares)."""
    d = NXY[None] - xy[:, None]                                  # (K, M, 2)
    axes = np.stack([np.broadcast_to(ang[:, None], d.shape[:2]),
                     np.broadcast_to(ang[:, None], d.shape[:2]) + np.pi / 2,
                     np.broadcast_to(NA[None], d.shape[:2]),
                     np.broadcast_to(NA[None], d.shape[:2]) + np.pi / 2], -1)   # (K,M,4)
    ux, uy = np.cos(axes), np.sin(axes)
    dist = np.abs(d[..., :1] * ux + d[..., 1:] * uy)
    r1 = half * (np.abs(np.cos(ang[:, None, None] - axes)) + np.abs(np.sin(ang[:, None, None] - axes)))
    r2 = half * (np.abs(np.cos(NA[None, :, None] - axes)) + np.abs(np.sin(NA[None, :, None] - axes)))
    return (dist - shrink * (r1 + r2)).max(-1)


def relax_sweep(placer, tile_w, board_w, board_h, step=0.06, angles=(2.0, 5.0),
                grout=0.1, gap_weight=4.0, align_weight=1.0, reach=0.6, seed=0,
                frozen_rows=()):
    """One greedy pass over all tiles (except those whose row is in frozen_rows,
    e.g. the regular brick background). Returns the number of tiles moved."""
    w, half = tile_w, tile_w / 2
    slack = placer.half / half
    sh = np.array([-step, 0.0, step]) * w
    da = np.radians([0.0, angles[0], -angles[0], angles[1], -angles[1]])
    order = [i for i, t in enumerate(placer.tiles) if t is not None and t[4] not in frozen_rows]
    np.random.default_rng(seed).shuffle(order)
    moved = 0
    for i in order:
        x, y, a, stage, row = placer.tiles[i]
        gx, gy = int(x // placer.cell), int(y // placer.cell)
        nb = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1)
              for k in placer.grid.get((gx + dx, gy + dy), ()) if k != i]
        if not nb:
            continue
        N = np.array([placer.tiles[k][:3] for k in nb])
        c, s = math.cos(a), math.sin(a)
        du, dv = np.meshgrid(sh, sh)
        du, dv = du.ravel(), dv.ravel()
        pos = np.stack([x + du * c - dv * s, y + du * s + dv * c], -1)          # (9, 2)
        P = np.repeat(pos, len(da), 0)
        A = np.tile(a + da, len(pos))                                          # (45,)
        inside = (P[:, 0] > half) & (P[:, 0] < board_w - half) & \
                 (P[:, 1] > half) & (P[:, 1] < board_h - half)
        inside[len(da) * 4] = True                                            # current pose
        ok = (_pair_clearance(P, A, N[:, :2], N[:, 2], half, slack) >= 0).all(1) & inside
        ok[len(da) * 4] = True                                                 # never forbid staying
        gap = _pair_clearance(P, A, N[:, :2], N[:, 2], half)                  # (45, M)
        close = gap < reach * w
        cnt = np.maximum(close.sum(1), 1)
        g_term = (np.where(close, ((gap - grout * w) / w) ** 2, 0)).sum(1) / cnt
        a_term = (np.where(close, (1 - np.cos(4 * (A[:, None] - N[None, :, 2]))) / 2, 0)).sum(1) / cnt
        score = gap_weight * g_term + align_weight * a_term
        score[~ok] = np.inf
        best = int(np.argmin(score))
        if best != len(da) * 4 and score[best] < score[len(da) * 4] - 1e-9:
            nx, ny, na = P[best, 0], P[best, 1], A[best]
            placer.grid[(gx, gy)].remove(i)
            placer.grid.setdefault((int(nx // placer.cell), int(ny // placer.cell)), []).append(i)
            placer.tiles[i] = (nx, ny, na, stage, row)
            moved += 1
    return moved


def snug_and_fill(placer, tile_w, board_w, board_h, rounds=8, fill_stage=-1, tol=0.01,
                  log=print, **kw):
    """Alternate relax sweeps and gap filling until convergence.
    Returns a list of (moved, added) per round."""
    hist = []
    for r in range(rounds):
        moved = relax_sweep(placer, tile_w, board_w, board_h, seed=r, **kw)
        added = fill_gaps(placer, tile_w, board_w, board_h, passes=1, stage=fill_stage, row=-1)
        n = sum(t is not None for t in placer.tiles)
        hist.append((moved, added))
        if log:
            log(f"  round {r + 1}: moved {moved} tiles, added {added} -> {n} tiles")
        if added == 0 and moved < tol * n:
            break
    return hist
