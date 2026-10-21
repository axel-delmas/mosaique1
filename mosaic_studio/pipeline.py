"""Run the steps, with previews saved in the project folder."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from . import colors, layout, render, segment
from .segment import no_progress, scaled


def preview_regions(project, selected=(), save=True):
    """Regions overlay. Cached per state of the regions and groups, so redrawing
    it for a new selection is quick; `save` also writes preview_1_regions.png."""
    lab_file = project.path("regions.npy")
    key = (lab_file, os.stat(lab_file).st_mtime_ns,
           json.dumps([(g["name"], g["members"], g["order"]) for g in project.groups]))
    im = render.regions_overlay(project.image(), project.regions(), project.ordered_groups(),
                                selected, cache_key=key)
    if save:
        im.save(project.path("preview_1_regions.png"))
    return im


def preview_lines(project, info):
    img = (project.image() * 255).astype(np.uint8)
    im = Image.fromarray((0.35 * img + 0.65 * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    board_w, board_h, ppc = project.board()
    cols = render.group_colors(len(project.groups))
    for gi, P, closed, kind in info["lines"]:
        if len(P) < 2:
            continue
        pts = [((x) * ppc, (board_h - y) * ppc) for x, y in P]
        if closed:
            pts.append(pts[0])
        d.line(pts, fill=tuple(int(v * 0.6) for v in cols[gi]), width=1)
    im.save(project.path("preview_2_lines.png"))
    return im


def render_mosaic(project, tiles, width_px=1800, crop=None):
    board_w, board_h, _ = project.board()
    return render.draw_tiles(tiles, board_w, board_h, width_px=width_px, crop=crop)


def build(project, log=print, progress=no_progress):
    """Steps 2-5 from the current regions and groups. Returns (tiles, mosaic image)."""
    t0 = time.time()
    log("Guide lines and tiles...")
    tiles, info = layout.build_layout(project, log, scaled(progress, 0, 0.75))
    progress(0.75, "drawing the guide lines")
    preview_lines(project, info)
    log("Colours...")
    progress(0.8, "choosing tile colours")
    colors.color_tiles(project, tiles, log)
    progress(0.9, "saving tiles.csv and the shopping list")
    counts = colors.export(project, tiles)
    progress(0.93, "drawing the mosaic")
    im = render_mosaic(project, tiles)
    im.save(project.path("mosaic.png"))
    json.dump([{k: (list(v) if isinstance(v, tuple) else v) for k, v in t.items()} for t in tiles],
              open(project.path("tiles.json"), "w"))
    board_w, board_h, _ = project.board()
    cover = len(tiles) * project.cfg["tile_cm"] ** 2 / (board_w * board_h)
    log(f"Done: {len(tiles):,} tiles, {len(counts)} colours, coverage {100 * cover:.0f}% "
        f"({time.time() - t0:.0f}s)")
    progress(1.0, "done")
    return tiles, im


OUTPUTS = ("mosaic.png", "tiles.csv", "shopping_list.csv")


def is_built(project):
    """True if the project folder holds a finished mosaic."""
    return all(os.path.exists(project.path(f)) for f in OUTPUTS)


def run_all(project, image_path=None, log=print, progress=no_progress):
    """Fully automatic: (new picture) -> segmentation -> default groups -> mosaic."""
    if image_path:
        project.set_image(image_path)
    if project.regions() is None or not project.groups:
        log("Segmenting (SAM)...")
        segment.segment_project(project, log, progress=scaled(progress, 0, 0.6))
    preview_regions(project)
    return build(project, log, scaled(progress, 0.6, 1))
