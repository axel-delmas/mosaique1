"""Steps 2-4: guide lines per group (in its fill style) and tile placement.

Every group gets:
  * outline rows: rows following its outline (and, for echo style, its inner
    lines, i.e. the borders between its member regions), `outline_rows` of them;
  * a fill in its style: echo (keep echoing inward), brick / grid / rows
    (straight rows at `angle`), fan (arcs around `fan_center`).
Groups are placed in their order (front first). Inside a group: outline rows
from the outside in, longest lines first, then the fill. Each line is laid in
one go; leftover pieces shorter than a few tiles are removed. Then optional gap
filling and greedy snugging.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from .geometry import (Placer, _length, _lookup, _runs, find_corners, line_tiles,
                       relax_sweep, resample, smooth_between, to_board)

FROZEN = 10_000          # row code for tiles of straight styles (not moved by snugging)


# --------------------------------------------------------------------------- #
# Barriers (outline + inner lines) and distance fields
# --------------------------------------------------------------------------- #
def boundary_of(mask):
    """Pixels of `mask` on its edge (next to a non-mask pixel or the image border)."""
    er = ndimage.binary_erosion(mask, border_value=0)
    return mask & ~er


def inner_lines(lab, members, detail):
    """Borders between the member regions of a group. detail in [0, 1]: 1 keeps
    the borders of all members, 0 none; in between, only members whose area is
    at least (1 - detail) x the largest member's area draw their outline."""
    if detail <= 0 or len(members) < 2:
        return np.zeros(lab.shape, bool)
    area = {m: int((lab == m).sum()) for m in members}
    big = max(area.values())
    keep = [m for m in members if area[m] >= (1 - detail) * big * 0.999]
    out = np.zeros(lab.shape, bool)
    for m in keep:
        out |= boundary_of(lab == m)
    return out


# --------------------------------------------------------------------------- #
# Line generation
# --------------------------------------------------------------------------- #
def level_lines(field, mask, levels, ppc, board_h, spacing):
    """Level curves of `field` (cm) at the given values, kept only inside mask.
    Returns list of (points (n, 2) cm, closed, level index)."""
    f = np.where(mask, field, -1.0)
    out = []
    for k, v in enumerate(levels):
        for c in find_contours(f, v):
            closed = np.allclose(c[0], c[-1])
            pts = to_board(c, ppc, board_h)
            if closed:
                pts = pts[:-1]
            keep = _lookup(mask, pts, ppc, board_h)
            for idx, cl in _runs(keep, closed):
                piece = pts[idx]
                if len(piece) >= 2 and _length(piece, cl) >= spacing:
                    out.append((piece, cl, k))
    return out


def straight_rows(zone, ppc, board_w, board_h, spacing, angle_deg, style):
    """Tile candidates on straight parallel rows at `angle_deg` across `zone`.
    brick: joints offset by half a tile every other row; grid: aligned joints;
    rows: tiles packed from the start of each run. Returns a list of
    (tiles, closed=False, row_key) with tiles = [(x, y, a, False, t)]."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    corners = np.array([[0, 0], [board_w, 0], [0, board_h], [board_w, board_h]])
    u = corners @ np.array([c, s])
    v = corners @ np.array([-s, c])
    step = spacing / 4
    us = np.arange(u.min() + step / 2, u.max(), step)
    lines = []
    for j in range(int(np.floor(v.min() / spacing)), int(np.ceil(v.max() / spacing)) + 1):
        vj = (j + 0.5) * spacing
        pts = np.stack([us * c - vj * s, us * s + vj * c], -1)
        ok = ((pts[:, 0] > 0) & (pts[:, 0] < board_w) & (pts[:, 1] > 0) & (pts[:, 1] < board_h))
        ok[ok] = _lookup(zone, pts[ok], ppc, board_h)
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for i0, i1 in zip(edges[::2], edges[1::2]):
            u0, u1 = us[i0], us[i1 - 1]
            if u1 - u0 < 0.5 * spacing:
                continue
            if style == "rows":
                n = int((u1 - u0 + step) // spacing)
                ucs = u0 - step / 2 + spacing / 2 + spacing * np.arange(n)
            else:
                shift = 0.5 * (j % 2) if style == "brick" else 0.0
                ucs = (np.arange(np.ceil(u0 / spacing - 0.5 - shift),
                                 np.floor(u1 / spacing - 0.5 - shift) + 1) + 0.5 + shift) * spacing
            tiles = [(uc * c - vj * s, uc * s + vj * c, a, False, float(uc)) for uc in ucs]
            if tiles:
                lines.append((tiles, False, j))
    return lines


def group_lines(g, gmask, barrier, ppc, board_w, board_h, spacing, corner_range=(60, 120),
                smooth=0.75):
    """All tile candidate lines of one group, in placement order.
    Returns a list of dicts: tiles, closed, row, kind, length."""
    step = spacing / 4
    dist = ndimage.distance_transform_edt(~barrier) / ppc
    if not gmask.any():
        return [], dist
    n_out = int(g["outline_rows"])
    style = g["style"]
    if style == "echo":
        n_out = int(np.ceil(dist[gmask].max() / spacing)) + 1
    raw = level_lines(dist, gmask, [(k + 0.5) * spacing for k in range(n_out)], ppc, board_h,
                      spacing)
    out = []
    for pts, closed, k in raw:
        P = resample(pts, closed, step)
        if len(P) < 3:
            continue
        cor = find_corners(P, closed, spacing, step, *corner_range)
        Ps = smooth_between(P, closed, cor, smooth * spacing / step)
        out.append(dict(tiles=line_tiles(Ps, closed, cor, spacing), P=Ps, closed=closed,
                        row=k, kind="outline", length=_length(Ps, closed)))
    if style != "echo":
        zone = gmask & (dist >= (n_out - 0.1) * spacing) if n_out else gmask
        if style in ("brick", "grid", "rows"):
            for tiles, closed, j in straight_rows(zone, ppc, board_w, board_h, spacing,
                                                  g["angle"], style):
                out.append(dict(tiles=tiles, P=np.array([t[:2] for t in tiles]), closed=False,
                                row=FROZEN, kind="fill", length=len(tiles) * spacing))
        elif style == "fan":
            cx, cy = g["fan_center"][0] * board_w, g["fan_center"][1] * board_h
            H, W = gmask.shape
            yy, xx = np.mgrid[0:H, 0:W]
            radial = np.hypot((xx + 0.5) / ppc - cx, board_h - (yy + 0.5) / ppc - cy)
            r0 = radial[zone].min() if zone.any() else 0
            levels = np.arange(np.floor(r0 / spacing) + 0.5, radial[zone].max() / spacing + 1) * spacing \
                if zone.any() else []
            for pts, closed, k in level_lines(radial, zone, list(levels), ppc, board_h, spacing):
                P = resample(pts, closed, step)
                if len(P) < 3:
                    continue
                Ps = smooth_between(P, closed, [], smooth * spacing / step)
                out.append(dict(tiles=line_tiles(Ps, closed, [], spacing), P=Ps, closed=closed,
                                row=n_out + k, kind="fill", length=_length(Ps, closed)))
    out.sort(key=lambda l: (l["kind"] == "fill", l["row"] if l["kind"] == "outline" else 0,
                            -l["length"]))
    return out, dist


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
def place_line(placer, tiles, closed, inside, gi, row, board_w, board_h, min_run):
    """Lay a line's tiles in order; drop pieces shorter than min_run tiles.
    Tiles whose centre is outside the group are treated as not fitting."""
    ids = []
    for (x, y, a, corner, _), ok in zip(tiles, inside):
        ids.append(placer.try_add(x, y, a, gi, board_w, board_h, row) if ok else -1)
    good = np.array([i >= 0 for i in ids])
    for idx, _ in _runs(good, closed):
        if len(idx) < min_run:
            for j in idx:
                placer.remove(ids[j])
    return int(good.sum())


def empty_space(tiles, tile_w, board_w, board_h, res):
    from skimage.draw import polygon
    H, W = int(np.ceil(board_h / res)), int(np.ceil(board_w / res))
    full = np.zeros((H, W), bool)
    h = tile_w / 2
    for x, y, a in tiles:
        c, s_ = math.cos(a), math.sin(a)
        pts = [(x + dx * c - dy * s_, y + dx * s_ + dy * c)
               for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        rr, cc = polygon([(board_h - py) / res for _, py in pts], [px / res for px, _ in pts], (H, W))
        full[rr, cc] = True
    full[0, :] = full[-1, :] = full[:, 0] = full[:, -1] = True
    return ndimage.distance_transform_edt(~full) * res


def fill_gaps(placer, gmap, ppc, tile_w, board_w, board_h, passes=2, min_clear=0.42):
    """Add tiles wherever a whole tile still fits, inside the group of that spot,
    turned like the nearest tile of the same group."""
    from scipy.spatial import cKDTree
    res = tile_w / 6
    turns = [0.0, 0.15, -0.15, 0.35, -0.35, math.pi / 4]
    shifts = [(0, 0), (0.12, 0), (-0.12, 0), (0, 0.12), (0, -0.12)]
    added = 0
    for _ in range(passes):
        cur = [t for t in placer.tiles if t is not None]
        if not cur:
            break
        T = np.array([t[:4] for t in cur])
        trees = {gi: (cKDTree(T[T[:, 3] == gi, :2]), T[T[:, 3] == gi, 2])
                 for gi in np.unique(T[:, 3]).astype(int)}
        d = empty_space(T[:, :3], tile_w, board_w, board_h, res)
        ys, xs = np.nonzero(d >= min_clear * tile_w)
        if len(ys) == 0:
            break
        before = added
        for i in np.argsort(-d[ys, xs]):
            x, y = (xs[i] + 0.5) * res, board_h - (ys[i] + 0.5) * res
            gi = int(_lookup(gmap, np.array([[x, y]]), ppc, board_h)[0])
            if gi < 0:
                continue
            if gi in trees:
                tree, ang = trees[gi]
                a0 = ang[tree.query([x, y])[1]]
            else:
                a0 = 0.0
            done = False
            for dx, dy in shifts:
                px, py = x + dx * tile_w, y + dy * tile_w
                if _lookup(gmap, np.array([[px, py]]), ppc, board_h)[0] != gi:
                    continue
                for da in turns:
                    if placer.try_add(px, py, a0 + da, gi, board_w, board_h, -1) >= 0:
                        added += 1
                        done = True
                        break
                if done:
                    break
        if added == before:
            break
    return added


def build_layout(project, log=print, progress=None):
    """Guide lines and tiles for the whole project.
    Returns (tiles: list of dicts, info: dict with lines per group etc.).
    progress(fraction, stage), if given, is told how far along it is."""
    progress = progress or (lambda frac, stage: None)
    cfg = project.cfg
    lab = project.regions()
    groups = project.ordered_groups()
    board_w, board_h, ppc = project.board()
    tile_w = float(cfg["tile_cm"])
    spacing = tile_w + float(cfg["grout_cm"])
    gmap = project.group_map()
    frame = np.zeros(lab.shape, bool)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    placer = Placer(tile_w, slack=min(0.97, 1 - 0.3 * cfg["grout_cm"] / tile_w))
    info = {"lines": [], "spacing": spacing}
    steps = len(groups) + 1 + int(cfg["snug_rounds"])
    for gi, g in enumerate(groups):
        progress(gi / steps, f"laying tiles in '{g['name']}' ({gi + 1} of {len(groups)})")
        gmask = gmap == gi
        barrier = boundary_of(gmask) | (frame & gmask)
        if g["style"] == "echo":
            barrier |= inner_lines(lab, g["members"], g["inner_detail"])
        lines, _ = group_lines(g, gmask, barrier, ppc, board_w, board_h, spacing)
        n = 0
        for ln in lines:
            pts = np.array([t[:2] for t in ln["tiles"]]) if ln["tiles"] else np.zeros((0, 2))
            inside = (_lookup(gmap, pts, ppc, board_h) == gi) if len(pts) else []
            n += place_line(placer, ln["tiles"], ln["closed"], inside, gi, ln["row"],
                            board_w, board_h, 2 if ln["row"] == FROZEN else 3)
            info["lines"].append((gi, ln["P"], ln["closed"], ln["kind"]))
        log(f"  {g['name']}: {g['style']}, {len(lines)} lines")
    if cfg["fill_gaps"]:
        progress(len(groups) / steps, "filling leftover gaps")
        added = fill_gaps(placer, gmap, ppc, tile_w, board_w, board_h)
        log(f"  gap filling: +{added} tiles")
    for r in range(int(cfg["snug_rounds"])):
        progress((len(groups) + 1 + r) / steps, f"snugging tiles (round {r + 1} of {cfg['snug_rounds']})")
        moved = relax_sweep(placer, tile_w, board_w, board_h, seed=r, frozen_rows=(FROZEN,))
        added = fill_gaps(placer, gmap, ppc, tile_w, board_w, board_h, passes=1) if cfg["fill_gaps"] else 0
        log(f"  snug round {r + 1}: moved {moved}, added {added}")
    tiles = [dict(x=t[0], y=t[1], angle=t[2], group=int(t[3]), row=int(t[4]), size=tile_w)
             for t in placer.tiles if t is not None]
    return tiles, info
