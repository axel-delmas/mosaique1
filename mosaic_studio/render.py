"""Previews and final renderings (numpy + PIL only, fast enough for the app)."""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


def group_colors(n):
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.65, 0.95))
            for i in range(n)]


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _inside_point(mask, step=3):
    """The most inside point of a (cropped) mask, as (x, y). The distance
    transform runs subsampled by `step`: the label only needs to be roughly
    central, and this is much faster."""
    d = ndimage.distance_transform_edt(np.pad(mask[::step, ::step], 1))[1:-1, 1:-1]
    y, x = np.unravel_index(d.argmax(), d.shape)
    return x * step, y * step


def label_point(mask):
    """Where a mask's label goes, as (x, y), working on its bounding box only."""
    sl = ndimage.find_objects(mask.astype(np.int8))
    if not sl or sl[0] is None:
        return 0, 0
    ys, xs = sl[0]
    x, y = _inside_point(mask[ys, xs])
    return xs.start + x, ys.start + y


_base_cache = {}


def _overlay_base(image, lab, groups):
    """Everything in the overlay that does not depend on the selection."""
    img = (image * 255).astype(np.uint8)
    cols = group_colors(len(groups))
    lut = np.zeros((lab.max() + 1, 3), np.uint8)
    gid = np.full(lab.max() + 1, -1)
    for i, g in enumerate(groups):
        for m in g["members"]:
            if m <= lab.max():
                lut[m] = cols[i]
                gid[m] = i
    tint = (0.55 * img + 0.45 * lut[lab]).astype(np.uint8)
    edge = np.zeros(lab.shape, bool)
    edge[:-1] |= lab[:-1] != lab[1:]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    gm = gid[lab]
    gedge = np.zeros(lab.shape, bool)
    gedge[:-1] |= gm[:-1] != gm[1:]
    gedge[:, :-1] |= gm[:, :-1] != gm[:, 1:]
    gedge = ndimage.binary_dilation(gedge, iterations=1)
    numbers = []
    for r, sl in enumerate(ndimage.find_objects(lab), 1):     # sl = bounding box of region r
        if sl is None:
            continue
        m = lab[sl] == r
        if m.sum() >= 30:
            x, y = _inside_point(m)
            numbers.append((r, (sl[1].start + x, sl[0].start + y)))
    names = []
    for i, g in enumerate(groups):
        m = np.isin(lab, g["members"])
        if m.any():
            names.append((i, g, label_point(m)))
    return dict(tint=tint, edge=edge, gedge=gedge, numbers=numbers, names=names, cols=cols)


def regions_overlay(image, lab, groups, selected=(), show_numbers=True, cache_key=None):
    """Picture tinted by group, region borders in black, group borders thick
    white, selected regions highlighted, region numbers and group names.
    With a cache_key (that changes whenever the regions or groups do), the
    part that does not depend on the selection is computed only once, so
    clicking regions stays fast."""
    base = _base_cache.get(cache_key) if cache_key is not None else None
    if base is None:
        base = _overlay_base(image, lab, groups)
        if cache_key is not None:
            _base_cache.clear()
            _base_cache[cache_key] = base
    out = base["tint"].copy()
    if len(selected):
        sel = np.isin(lab, list(selected))
        out[sel] = (0.35 * out[sel] + 0.65 * np.array([255, 255, 255])).astype(np.uint8)
    out[base["edge"]] = 20
    out[base["gedge"]] = 255
    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    s = max(12, int(max(lab.shape) / 70))
    f, fg = _font(s), _font(int(s * 1.3))
    if show_numbers:
        for r, (x, y) in base["numbers"]:
            d.text((x, y), str(r), fill="white", font=f, anchor="mm", stroke_width=3,
                   stroke_fill="black")
    for i, g, (x, y) in base["names"]:
        d.text((x, y + int(1.4 * s)), f"{g['order'] + 1}. {g['name']}", fill=base["cols"][i],
               font=fg, anchor="mm", stroke_width=4, stroke_fill="black")
    return pil


def draw_tiles(tiles, board_w, board_h, width_px=1600, grout=(215, 210, 200), outline=None,
               tile_colors=None, crop=None):
    """Render tiles (list of dicts x, y, angle(rad), color (r,g,b 0-255)) to a PIL image.
    crop = (x0, x1, y0, y1) in cm to zoom."""
    x0, x1, y0, y1 = crop if crop else (0, board_w, 0, board_h)
    s = width_px / (x1 - x0)
    W, H = width_px, int(round((y1 - y0) * s))
    im = Image.new("RGB", (W, H), grout)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(tiles):
        h = t["size"] / 2
        c, sn = np.cos(t["angle"]), np.sin(t["angle"])
        pts = [((t["x"] + dx * c - dy * sn - x0) * s, (y1 - (t["y"] + dx * sn + dy * c)) * s)
               for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        col = tuple(int(v) for v in (tile_colors[i] if tile_colors is not None else t["color"]))
        d.polygon(pts, fill=col, outline=outline)
    return im
