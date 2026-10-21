"""Geometry shared by all fill styles: polylines along which tiles are laid,
~90° corner detection and smoothing, exact overlap test between rotated squares,
a placement structure with a spatial hash, and the greedy "snug" relaxation.
(Adapted from the guide_lines / snug experiments.)"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

def to_board(rc, ppc, board_h):
    """(row, col) pixel coordinates -> (x, y) board coordinates in cm."""
    return np.stack([(rc[:, 1] + 0.5) / ppc, board_h - (rc[:, 0] + 0.5) / ppc], -1)


def _lookup(M, pts, ppc, board_h):
    """Value of pixel map M at board points (nearest pixel)."""
    H, W = M.shape
    r = np.clip(((board_h - pts[:, 1]) * ppc - 0.5).astype(int), 0, H - 1)
    c = np.clip((pts[:, 0] * ppc - 0.5).astype(int), 0, W - 1)
    return M[r, c]


def _runs(keep, closed):
    """Index runs where keep is True (wrapping around for closed lines).
    Returns a list of (index array, still_closed)."""
    n = len(keep)
    if keep.all():
        return [(np.arange(n), closed)]
    if not keep.any():
        return []
    if closed:                                  # start right after a dropped point
        start = int(np.flatnonzero(~keep)[0])
        order = (np.arange(n) + start) % n
    else:
        order = np.arange(n)
    k = keep[order]
    edges = np.flatnonzero(np.diff(np.r_[0, k.astype(int), 0]))
    return [(order[a:b], False) for a, b in zip(edges[::2], edges[1::2])]


def _length(pts, closed):
    seg = np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0)
    return np.linalg.norm(seg, axis=1).sum()


def resample(pts, closed, step):
    """Points every `step` cm along the polyline."""
    P = np.vstack([pts, pts[:1]]) if closed else pts
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    L = s[-1]
    n = max(2, int(round(L / step)))
    t = np.linspace(0, L, n, endpoint=not closed)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], -1)


def _turns(P, closed):
    """Signed turning angle at each vertex (0 at open ends)."""
    if closed:
        a = np.roll(P, 1, 0); b = np.roll(P, -1, 0)
    else:
        a = np.vstack([P[:1], P[:-1]]); b = np.vstack([P[1:], P[-1:]])
    d1, d2 = P - a, b - P
    ang = np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(1))
    if not closed:
        ang[0] = ang[-1] = 0
    return ang


def find_corners(P, closed, spacing, step, lo=60, hi=120):
    """Indices of sharp ~90° turns: the turn summed over ±½ tile is between lo°
    and hi°, it is a local maximum, and at least 60% of it happens within ±1
    sample (a real corner, not a gentle curve)."""
    turn = _turns(P, closed)
    w = max(1, int(round(0.5 * spacing / step)))
    mode = "wrap" if closed else "constant"
    win = ndimage.uniform_filter1d(turn, 2 * w + 1, mode=mode) * (2 * w + 1)
    tight = ndimage.uniform_filter1d(turn, 3, mode=mode) * 3
    mag = np.abs(win)
    peak = mag == ndimage.maximum_filter1d(mag, 2 * w + 1, mode=mode)
    ok = peak & (mag >= math.radians(lo)) & (mag <= math.radians(hi)) & \
        (np.abs(tight) >= 0.6 * mag)
    idx = np.flatnonzero(ok)
    # keep one per cluster (plateaus of equal maxima)
    keep = []
    for i in idx:
        if not keep or i - keep[-1] > w:
            keep.append(i)
    if closed and len(keep) > 1 and keep[0] + len(P) - keep[-1] <= w:
        keep.pop()
    if not closed:
        keep = [i for i in keep if w <= i < len(P) - w]
    return keep


def smooth_between(P, closed, corners, sigma_samples):
    """Gaussian smoothing of the line, with corners (and open ends) held fixed:
    each stretch between fixed points is smoothed on its own."""
    if closed and not corners:
        return np.stack([ndimage.gaussian_filter1d(P[:, k], sigma_samples, mode="wrap")
                         for k in (0, 1)], -1)
    Q = P.copy()
    if closed:
        order = corners + [corners[0] + len(P)]
        for a, b in zip(order[:-1], order[1:]):
            idx = np.arange(a, b + 1) % len(P)
            Q[idx[1:-1]] = _smooth_open(P[idx], sigma_samples)[1:-1]
    else:
        stops = [0] + corners + [len(P) - 1]
        for a, b in zip(stops[:-1], stops[1:]):
            Q[a + 1:b] = _smooth_open(P[a:b + 1], sigma_samples)[1:-1]
    return Q


def _smooth_open(S, sigma):
    """Smooth an open stretch keeping both endpoints: smooth the deviation from
    the straight chord between them (so ends don't shrink inward)."""
    if len(S) < 3:
        return S
    t = np.linspace(0, 1, len(S))[:, None]
    chord = S[0] * (1 - t) + S[-1] * t
    dev = S - chord
    dev = np.stack([ndimage.gaussian_filter1d(dev[:, k], sigma, mode="constant")
                    for k in (0, 1)], -1)
    return chord + dev


def _tangent_angle(P, i, closed):
    n = len(P)
    a = P[(i - 1) % n] if (closed or i > 0) else P[i]
    b = P[(i + 1) % n] if (closed or i < n - 1) else P[i]
    d = b - a
    return math.atan2(d[1], d[0])


def _mean_angle4(a, b):
    """4-way average of two angles (a square turned by the result has sides
    along both directions when they are ~90° apart)."""
    return math.atan2(math.sin(4 * a) + math.sin(4 * b), math.cos(4 * a) + math.cos(4 * b)) / 4


def line_tiles(P, closed, corners, spacing):
    """Tiles (x, y, angle, is_corner, t) along one resampled, smoothed line,
    t = arc-length position, so tiles can be sorted along the line."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P,
                                                   axis=0), axis=1))]
    n_pts = len(P)
    L_total = s[-1]
    tiles = []

    def point_at(t):
        t = t % L_total if closed else min(max(t, 0), s[n_pts - 1])
        i = int(np.searchsorted(s, t, side="right") - 1)
        i = min(i, len(s) - 2)
        f = (t - s[i]) / max(s[i + 1] - s[i], 1e-9)
        a, b = P[i % n_pts], P[(i + 1) % n_pts]
        p = a + f * (b - a)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        return p, ang

    for c in corners:
        a_in = _tangent_angle(P, (c - 1) % n_pts, closed)
        a_out = _tangent_angle(P, (c + 1) % n_pts, closed)
        tiles.append((P[c][0], P[c][1], _mean_angle4(a_in, a_out), True, s[c]))

    def fill(t0, t1, start_fixed, end_fixed):
        L = t1 - t0
        if start_fixed and end_fixed:
            n = max(1, int(round(L / spacing)))
            ts = [t0 + m * L / n for m in range(1, n)]
        elif start_fixed:
            avail = L - spacing / 2
            n = int(round(avail / spacing)) if avail > 0 else 0
            ts = [t0 + m * avail / n for m in range(1, n + 1)] if n else []
        elif end_fixed:
            avail = L - spacing / 2
            n = int(round(avail / spacing)) if avail > 0 else 0
            ts = [t1 - m * avail / n for m in range(1, n + 1)] if n else []
        else:
            avail = L - spacing
            if avail < -0.4 * spacing:
                ts = []
            elif avail <= 0:
                ts = [(t0 + t1) / 2]
            else:
                n = int(round(avail / spacing))
                ts = [t0 + spacing / 2 + m * avail / max(n, 1) for m in range(n + 1)] \
                    if n else [(t0 + t1) / 2]
        for t in ts:
            p, ang = point_at(t)
            tiles.append((p[0], p[1], ang, False, t % L_total if closed else t))

    cs = [s[c] for c in corners]
    if closed:
        if not cs:
            n = max(1, int(round(L_total / spacing)))
            for m in range(n):
                p, ang = point_at(m * L_total / n)
                tiles.append((p[0], p[1], ang, False, m * L_total / n))
        else:
            for a, b in zip(cs, cs[1:] + [cs[0] + L_total]):
                fill(a, b, True, True)
    else:
        stops = [0.0] + cs + [s[n_pts - 1]]
        fixed = [False] + [True] * len(cs) + [False]
        for (a, fa), (b, fb) in zip(zip(stops[:-1], fixed[:-1]), zip(stops[1:], fixed[1:])):
            fill(a, b, fa, fb)
    tiles.sort(key=lambda q: q[4])
    return tiles


def squares_overlap(c, a, C, A, half):
    """Does square (center c, angle a) overlap any of squares (C (m,2), A (m,))?
    All have half-size `half`. Exact SAT test on the 4 candidate axes."""
    if len(C) == 0:
        return False
    d = C - c
    angles = np.stack([np.full(len(A), a), np.full(len(A), a + np.pi / 2), A, A + np.pi / 2], 1)
    ux, uy = np.cos(angles), np.sin(angles)
    dist = np.abs(d[:, :1] * ux + d[:, 1:] * uy)
    r1 = half * (np.abs(np.cos(a - angles)) + np.abs(np.sin(a - angles)))
    r2 = half * (np.abs(np.cos(A[:, None] - angles)) + np.abs(np.sin(A[:, None] - angles)))
    separated = (dist >= r1 + r2).any(1)
    return bool((~separated).any())


class Placer:
    """Placement with a spatial hash; a tile is accepted only if it does not
    overlap any placed tile (tested with a slightly shrunk square, `slack`, so
    that grazing corners are tolerated). Tiles can be removed again."""

    def __init__(self, tile_w, slack=0.92):
        self.w, self.half = tile_w, tile_w * slack / 2
        self.cell = tile_w * math.sqrt(2)
        self.grid = {}
        self.tiles = []          # (x, y, angle, stage, row) or None once removed

    def try_add(self, x, y, ang, stage, board_w, board_h, row=-1):
        """Index of the new tile, or -1 if it does not fit."""
        if not (0 < x < board_w and 0 < y < board_h):
            return -1
        gx, gy = int(x // self.cell), int(y // self.cell)
        near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for k in self.grid.get((gx + dx, gy + dy), ())]
        if near:
            T = np.array([self.tiles[k][:3] for k in near])
            if squares_overlap(np.array([x, y]), ang, T[:, :2], T[:, 2], self.half):
                return -1
        self.grid.setdefault((gx, gy), []).append(len(self.tiles))
        self.tiles.append((x, y, ang, stage, row))
        return len(self.tiles) - 1

    def remove(self, k):
        x, y = self.tiles[k][:2]
        self.grid[(int(x // self.cell), int(y // self.cell))].remove(k)
        self.tiles[k] = None

    def placed(self):
        return [t for t in self.tiles if t is not None]

    def add_line(self, tiles, closed, stage_of, row, board_w, board_h, min_run):
        """Place a whole line's tiles in order along it; afterwards drop every run
        of consecutive placed tiles shorter than `min_run` (no floating bits).
        Returns (placed, dropped) counts."""
        ids = [self.try_add(x, y, a, stage_of(c), board_w, board_h, row)
               for x, y, a, c, _ in tiles]
        ok = np.array([i >= 0 for i in ids])
        placed = 0
        for idx, _ in _runs(ok, closed):
            if len(idx) < min_run:
                for j in idx:
                    self.remove(ids[j])
            else:
                placed += len(idx)
        return placed, len(tiles) - placed


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
