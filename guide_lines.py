"""Mosaic built on continuous guide lines (andamento), like a mosaicist works.

1. contours      edge-map skeleton (cleaned) + the board edge
2. guide lines   echo rows: curves at distance (k + ½)·s from the contours
                 (s = tile + grout). How many rows depends on the side:
                 inside objects (regions enclosed by contours) `inside_rows`,
                 on the background side (regions touching the image border)
                 `outside_rows`, along the board edge `frame_rows`. By default
                 objects are filled entirely with echo rows. Straight horizontal
                 rows in brick bond fill the background further away.
3. corners       along every line, a sharp turn of 60°–120° concentrated
                 within ~½ tile is kept as a CORNER; all other bends are
                 smoothed so tiles never have to turn hard
4. tiles         corner tiles are placed first, sides along both legs; then
                 each stretch between corners / line ends gets evenly spaced
                 tiles following the line; order: corners, contour rows (inner
                 to outer), background. A tile that would overlap an
                 already-placed one (exact rotated-square test) is skipped.
5. colors        mean image color under each tile

Everything geometric is in cm; the edge map / distance field is in pixels of
the image, stretched over the board.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from edge_field import add_frame, normalize, prune, remove_spurs, skeleton, threshold
from mosaic import Mosaic, Tile


# --------------------------------------------------------------------------- #
# 1. Contours and distance
# --------------------------------------------------------------------------- #
def contour_skeleton(edge_map, board_w, threshold_value=0.3, spur_cm=0.5,
                     min_length_cm=1.0, hole_cm2=0.25, border_px=20, frame=True):
    """Cleaned 1-px contour lines (+ board edge) and pixels per cm."""
    h, w = edge_map.shape
    ppc = w / board_w
    e = normalize(edge_map, border_px)
    sk = skeleton(threshold(e, threshold_value), hole_cm2 * ppc ** 2)
    sk = prune(remove_spurs(sk, spur_cm * ppc), min_length_cm * ppc)
    if frame:
        sk = add_frame(sk, 0)
    return sk, ppc


def to_board(rc, ppc, board_h):
    """(row, col) pixel coordinates -> (x, y) board coordinates in cm."""
    return np.stack([(rc[:, 1] + 0.5) / ppc, board_h - (rc[:, 0] + 0.5) / ppc], -1)


def object_mask(sk, ppc, close_cm=0.3):
    """True inside objects, False on the background.

    The contours are thickened by `close_cm` to close small gaps, the rest of the
    image is split into connected regions, and every region touching the image
    border is background (wall, floor). Regions enclosed by contours are objects.
    Pixels on the thickened contours take the label of the nearest region."""
    r = max(1, int(round(close_cm * ppc)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    walls = ndimage.binary_dilation(sk, xx ** 2 + yy ** 2 <= r * r)
    lab, _ = ndimage.label(~walls)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, border[border > 0])
    _, (iy, ix) = ndimage.distance_transform_edt(walls, return_indices=True)
    return ~bg[iy, ix]


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


# --------------------------------------------------------------------------- #
# 2. Guide lines
# --------------------------------------------------------------------------- #
def echo_lines(dist_cm, spacing, rows_map, ppc, board_h):
    """Level curves of the distance field at (k + ½)·spacing.

    rows_map (pixels) says how many echo rows are allowed at each place
    (e.g. 3 inside objects, 1 on the background side, 3 along the frame); the
    parts of a level curve beyond that are dropped, which can split a curve.
    Returns a list of (points (n, 2) in cm, closed?, row index k)."""
    out = []
    for k in range(int(rows_map.max())):
        for c in find_contours(dist_cm, (k + 0.5) * spacing):
            closed = np.allclose(c[0], c[-1])
            pts = to_board(c, ppc, board_h)
            if closed:
                pts = pts[:-1]
            keep = _lookup(rows_map, pts, ppc, board_h) > k
            for idx, cl in _runs(keep, closed):
                piece = pts[idx]
                if len(piece) >= 2 and _length(piece, cl) >= spacing:
                    out.append((piece, cl, k))
    return out


def background_lines(dist_cm, spacing, rows_map, ppc, board_w, board_h, step, row_id):
    """Straight horizontal rows wherever the distance to the nearest contour is
    at least (allowed rows there - 0.1)·spacing. They are placed last, so any
    tile that would overlap an echo-row tile is rejected; this lets them fill
    right up to the echo rows instead of leaving an empty band."""
    out = []
    clear = (rows_map - 0.1) * spacing
    xs = np.arange(step / 2, board_w, step)
    for y in np.arange(spacing / 2, board_h, spacing):
        pts = np.stack([xs, np.full_like(xs, y)], -1)
        ok = _lookup(dist_cm, pts, ppc, board_h) >= _lookup(clear, pts, ppc, board_h)
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if (b - a) * step >= spacing:
                out.append((pts[a:b], False, row_id))
    return out


# --------------------------------------------------------------------------- #
# 3. Resample, corners, smoothing
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 4. Tiles along lines
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Overlap test between rotated squares (separating axis theorem)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Gap filling
# --------------------------------------------------------------------------- #
def empty_space(tiles, tile_w, board_w, board_h, res):
    """Distance (cm) from each point of a fine raster to the nearest tile or board
    edge. Pixels are `res` cm."""
    from skimage.draw import polygon
    H, W = int(np.ceil(board_h / res)), int(np.ceil(board_w / res))
    full = np.zeros((H, W), bool)
    for x, y, a in tiles:
        c, s_ = math.cos(a), math.sin(a)
        h = tile_w / 2
        pts = [(x + dx * c - dy * s_, y + dx * s_ + dy * c)
               for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        rr, cc = polygon([(board_h - py) / res for _, py in pts],
                         [px / res for px, _ in pts], full.shape)
        full[rr, cc] = True
    full[0, :] = full[-1, :] = full[:, 0] = full[:, -1] = True
    return ndimage.distance_transform_edt(~full) * res


def fill_gaps(placer, tile_w, board_w, board_h, passes=3, min_clear=0.42,
              stage=-1, row=-1):
    """Add tiles wherever a whole tile still fits between the placed ones.

    Candidate spots are the points furthest from any tile (largest empty space
    first). At each spot a tile is tried at the angle of its nearest placed
    neighbour (so it continues the local rows), then slightly turned, then
    slightly shifted; the first one that does not overlap is kept.
    Returns the number of tiles added."""
    from scipy.spatial import cKDTree
    res = tile_w / 6
    turns = [0.0, 0.15, -0.15, 0.35, -0.35, math.pi / 4]
    shifts = [(0, 0)] + [(dx * tile_w, dy * tile_w) for dx, dy in
                         ((0.12, 0), (-0.12, 0), (0, 0.12), (0, -0.12))]
    added = 0
    for _ in range(passes):
        cur = [t for t in placer.tiles if t is not None]
        if not cur:
            break
        T = np.array([t[:3] for t in cur])
        tree = cKDTree(T[:, :2])
        d = empty_space(T, tile_w, board_w, board_h, res)
        ys, xs = np.nonzero(d >= min_clear * tile_w)
        if len(ys) == 0:
            break
        order = np.argsort(-d[ys, xs])
        n_before = added
        for i in order:
            x, y = (xs[i] + 0.5) * res, board_h - (ys[i] + 0.5) * res
            a0 = T[tree.query([x, y])[1], 2]
            done = False
            for dx, dy in shifts:
                for da in turns:
                    if placer.try_add(x + dx, y + dy, a0 + da, stage, board_w, board_h, row) >= 0:
                        added += 1
                        done = True
                        break
                if done:
                    break
        if added == n_before:
            break
    return added


# --------------------------------------------------------------------------- #
# Everything
# --------------------------------------------------------------------------- #
def bg_stage_fill(echo_rows):
    """Stage id given to gap-filling tiles (after the background)."""
    return echo_rows + 2


def build_line_mosaic(edge_map, image, board_w, tile_w, spacing_factor=1.1,
                      inside_rows=None, outside_rows=1, frame_rows=3, close_cm=0.3,
                      min_run=3, fill_gaps_pass=True, snug_rounds=8,
                      smooth=0.75, corner_range=(60, 120), return_steps=False,
                      **contour_kw):
    """inside_rows / outside_rows: echo rows on the object / background side of a
    contour (inside_rows=None: keep echoing inward until the object is full, no
    grid inside objects); frame_rows: rows along the board edge. The background
    grid is laid in brick bond (every other row shifted by half a tile). Objects vs
    background come from object_mask (regions not connected to the image border).

    snug_rounds: then alternate greedy "snugging" (small shifts / turns that make
    each tile fit and align better, see snug.py) with gap filling, up to this many
    rounds or until nothing changes (0 = off).
    fill_gaps_pass: afterwards, add tiles wherever a whole tile still fits
    (fill_gaps), turned like their nearest neighbour.

    Placement goes from the outside inwards: all first rows (along every contour
    and the frame), then all second rows, and so on, the background last. Within
    a row, longer lines go first. Each line is laid in one go along its length,
    and any piece of it that ends up shorter than `min_run` tiles (because other
    tiles were already there) is removed, so there are no isolated tiles.
    """
    import torch
    from deformed_grid import mean_colors

    H, W = edge_map.shape
    board_h = board_w * H / W
    spacing = spacing_factor * tile_w
    step = spacing / 4
    sk_c, ppc = contour_skeleton(edge_map, board_w, frame=False, **contour_kw)
    sk = add_frame(sk_c, 0)                           # contours + board edge
    dist_cm = ndimage.distance_transform_edt(~sk) / ppc
    d_obj = ndimage.distance_transform_edt(~sk_c) / ppc if sk_c.any() else np.full(sk.shape, np.inf)
    objects = object_mask(sk_c, ppc, close_cm)
    fill = inside_rows is None
    if fill:                                          # fill objects completely with echo rows
        inside_rows = int(np.ceil(dist_cm[objects].max() / spacing)) + 1 if objects.any() else 0
    rows_map = np.where(dist_cm < d_obj - 1e-9, frame_rows,          # nearest line = frame
                        np.where(objects, inside_rows, outside_rows))
    echo_rows = int(rows_map.max())                   # row id used for the background grid
    bg_rows = np.where(objects, 1e6, rows_map) if fill else rows_map  # no grid inside objects

    raw = [(p, c, k, "echo") for p, c, k in echo_lines(dist_cm, spacing, rows_map, ppc, board_h)]
    raw += [(p, c, k, "background") for p, c, k in
            background_lines(dist_cm, spacing, bg_rows, ppc, board_w, board_h, step, echo_rows)]
    lines = []
    for pts, closed, k, kind in raw:
        P = resample(pts, closed, step)
        if len(P) < 3:
            continue
        cor = [] if kind == "background" else find_corners(P, closed, spacing, step, *corner_range)
        Ps = smooth_between(P, closed, cor, smooth * spacing / step)
        lines.append(dict(P=Ps, raw=P, closed=closed, corners=cor, row=k, kind=kind,
                          length=_length(Ps, closed)))

    # outside -> inside: by row, longest lines first, background last
    lines.sort(key=lambda l: (l["kind"] == "background", l["row"], -l["length"]))
    placer = Placer(tile_w)
    skipped = np.zeros(echo_rows + 2, int)
    n_cand = 0
    for ln in lines:
        if ln["kind"] == "background":             # tiles on shared columns, brick bond
            x0, x1, y = ln["P"][0, 0], ln["P"][-1, 0], ln["P"][0, 1]
            shift = 0.5 * (int(round(y / spacing - 0.5)) % 2)
            tl = [((i + 0.5 + shift) * spacing, y, 0.0, False, 0.0)
                  for i in range(int(np.ceil(x0 / spacing - 0.5 - shift)),
                                 int(np.floor(x1 / spacing - 0.5 - shift)) + 1)]
            run = 2
        else:
            tl = line_tiles(ln["P"], ln["closed"], ln["corners"], spacing)
            run = min_run
        stage = 1 + ln["row"]
        _, dropped = placer.add_line(tl, ln["closed"] and ln["kind"] != "background",
                                     lambda c, st=stage: 0 if c else st, ln["row"],
                                     board_w, board_h, run)
        skipped[min(stage, len(skipped) - 1)] += dropped
        n_cand += len(tl)
    n_filled = fill_gaps(placer, tile_w, board_w, board_h, stage=bg_stage_fill(echo_rows),
                         row=-1) if fill_gaps_pass else 0
    snug_hist = []
    if snug_rounds:
        from snug import snug_and_fill
        snug_hist = snug_and_fill(placer, tile_w, board_w, board_h, rounds=snug_rounds,
                                  fill_stage=bg_stage_fill(echo_rows),
                                  frozen_rows=(echo_rows,))   # keep the brick wall regular
    placer.tiles = placer.placed()

    T = np.array([t[:3] for t in placer.tiles])
    cols = mean_colors(T[:, :2], T[:, 2], image, tile_w, board_w, board_h)
    tiles = [Tile(color=tuple(int(round(v * 255)) for v in np.clip(c, 0, 1)),
                  position=(x, y), orientation=math.degrees(a) % 360)
             for (x, y, a), c in zip(T, cols)]
    m = Mosaic(board_w, board_h, tile_w, tiles)
    if not return_steps:
        return m
    stages = np.array([t[3] for t in placer.tiles])
    rows = np.array([t[4] for t in placer.tiles])      # echo row (echo_rows = background)
    return m, dict(skeleton=sk, dist_cm=dist_cm, lines=lines, stages=stages, rows=rows,
                   echo_rows=echo_rows, objects=objects, rows_map=rows_map,
                   skipped=skipped, candidates=n_cand, n_filled=n_filled, snug_hist=snug_hist, ppc=ppc, board_h=board_h,
                   spacing=spacing)
