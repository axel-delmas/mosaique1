"""Hand-drawn shapes: the outline of a region for the editor, and the edits
made with it (a new shape, or a reshaped region).

The editor (editor.js) works with closed outlines ("loops") in image pixel
coordinates: x to the right, y down, pixel (row r, col c) covering
[c, c+1) x [r, r+1), so its centre is (c + 0.5, r + 0.5). A shape is the
even-odd fill of its loops, so a loop inside another one is a hole.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours

from . import groups as G


def image_data_url(project, quality=90):
    """The working picture as an inline JPEG for the editor (no file serving needed)."""
    buf = io.BytesIO()
    Image.open(project.path("image.png")).convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def densify(P, step):
    """Closed polyline with points added so that none are more than `step` apart."""
    out = []
    for a, b in zip(P, np.roll(P, -1, 0)):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.extend(a + (b - a) * k / n for k in range(n))
    return np.array(out)


def region_outline(lab, r, step=6.0, min_area=30, tolerance=0.6):
    """Loops (lists of [x, y]) outlining region r, holes included. Pieces smaller
    than min_area pixels are left out; points closer than `tolerance` pixels to
    a straight line are dropped (keeps the editor light, not the shape), then
    points are added along long straight stretches so no two are more than
    `step` apart. The editor cuts outlines only at existing points, so an
    outline it sends back untouched is exactly the one it got."""
    m = lab == r
    sl = ndimage.find_objects(m.astype(np.int8))
    if not sl or sl[0] is None:
        return []
    ys, xs = sl[0]
    crop = np.pad(m[ys, xs], 1).astype(np.float32)
    loops = []
    for c in find_contours(crop, 0.5):
        c = approximate_polygon(c, tolerance)[:-1]            # closed: drop the repeated point
        if len(c) < 3:
            continue
        x, y = c[:, 1] - 1 + xs.start + 0.5, c[:, 0] - 1 + ys.start + 0.5
        if 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < min_area:
            continue
        loops.append(np.round(densify(np.stack([x, y], 1), step), 2).tolist())
    return loops


def rasterize(loops, shape):
    """(H, W) mask of the even-odd fill of the loops (PIL's scanline fill: fast)."""
    mask = np.zeros(shape, bool)
    for loop in loops:
        if len(loop) < 3:
            continue
        im = Image.new("1", (shape[1], shape[0]), 0)
        ImageDraw.Draw(im).polygon([(x - 0.5, y - 0.5) for x, y in loop], fill=1)
        mask ^= np.asarray(im, bool)
    return mask


def edited_mask(lab, r, before, after, margin=2):
    """Region r after the user turned outline `before` into `after`. Where the
    drawing changed (plus `margin` pixels, which covers the old edge), the
    region is exactly the new drawing; everywhere else it keeps its exact
    pixels, so the untouched parts of the outline do not move at all."""
    A, B = rasterize(after, lab.shape), rasterize(before, lab.shape)
    zone = ndimage.binary_dilation(A ^ B, iterations=margin)
    return np.where(zone, A, lab == r)


def absorb_specks(lab, where, min_px=30, pad=40):
    """Pieces of regions smaller than min_px that touch `where` (the area an
    edit changed) join the region around them: no stray pixels or 1-pixel
    slivers are left along an old or new edge. Works in place on the edit's
    bounding box; returns lab."""
    ys, xs = np.nonzero(where)
    if not len(ys):
        return lab
    H, W = lab.shape
    box = (slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)),
           slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)))
    sub, near = lab[box], where[box]
    speck = np.zeros(sub.shape, bool)
    for r in np.unique(sub[near]):
        cc, n = ndimage.label(sub == r)
        small = np.bincount(cc.ravel(), minlength=n + 1) < min_px
        small[0] = False
        edge = np.unique(np.r_[cc[0], cc[-1], cc[:, 0], cc[:, -1]])    # may continue outside the box
        small[edge] = False
        small[np.setdiff1d(np.arange(n + 1), np.unique(cc[near]))] = False
        speck |= small[cc]
    if speck.any() and not speck.all():
        _, (iy, ix) = ndimage.distance_transform_edt(speck, return_indices=True)
        sub[speck] = sub[iy[speck], ix[speck]]
    return lab


def add_shape(project, mask, name, kind="object"):
    """A drawn shape becomes a new region (taking its pixels from whatever was
    there) in a new group of its own, at the front. Returns the group's name."""
    lab = project.regions()
    new = int(lab.max()) + 1
    before = lab.copy()
    lab[mask] = new
    absorb_specks(lab, ndimage.binary_dilation(lab != before, iterations=2))
    G.drop_missing(project, lab)
    project.set_regions(lab)
    return G.new_group_from(project, [new], name, kind)


def reshape_region(project, r, mask):
    """Region r becomes exactly `mask`: it takes the pixels it now covers from its
    neighbours, and the pixels it no longer covers go to the nearest other region."""
    lab = project.regions()
    before = lab.copy()
    freed = (lab == r) & ~mask
    lab[mask] = r
    if freed.any():
        others = lab != r
        if others.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~others, return_indices=True)
            lab[freed] = lab[iy[freed], ix[freed]]
    absorb_specks(lab, ndimage.binary_dilation(lab != before, iterations=2))
    G.drop_missing(project, lab)
    project.set_regions(lab)
    project.save()
