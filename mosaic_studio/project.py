"""A project = one picture + all the choices made about it, saved in a folder.

    <folder>/image.png          working copy of the picture (resized)
    <folder>/project.json       settings and region groups (human readable)
    <folder>/regions.npy        segmentation: label map of small regions
    <folder>/preview_*.png      previews of each step
    <folder>/history/           undo / redo steps (history.py)
    <folder>/mosaic.png, tiles.csv, shopping_list.csv   final outputs
"""
from __future__ import annotations

import copy
import json
import os
import shutil

import numpy as np
from PIL import Image

STYLES = ["echo", "brick", "grid", "rows", "fan"]
STYLE_HELP = {
    "echo": "rows following the outline and inner lines, all the way in",
    "brick": "straight rows, joints offset by half a tile (angle adjustable)",
    "grid": "straight rows with aligned joints (angle adjustable)",
    "rows": "straight rows at the angle, tiles placed freely along each row",
    "fan": "concentric arcs around a centre point (opus circumactum)",
}

DEFAULTS = {
    "board_width_cm": 30.0,
    "tile_cm": 0.4,
    "grout_cm": 0.04,
    "working_size": 1200,          # longest side of the working image, pixels
    "sam_model": "vit_l",          # vit_l (better) or vit_b (faster, smaller download)
    "sam_points_per_side": 32,
    "min_region_pct": 0.3,         # regions smaller than this % of the image are merged
    "background_min_pct": 3.0,     # border-touching regions at least this big = background
    "smooth_image": True,
    "saturation": 1.15,
    "contrast": 12.0,              # lightness push of outline tiles at group borders
    "fill_gaps": True,
    "snug_rounds": 0,
}

GROUP_DEFAULTS = {
    "object": {"style": "echo", "angle": 0.0, "outline_rows": 1, "inner_detail": 1.0,
               "colors": 8, "fan_center": [0.5, 0.0]},
    "background": {"style": "brick", "angle": 0.0, "outline_rows": 1, "inner_detail": 0.0,
                   "colors": 6, "fan_center": [0.5, 0.0]},
}


def new_group(gid, name, kind, members, order):
    g = {"id": gid, "name": name, "kind": kind, "members": sorted(int(m) for m in members),
         "order": order}
    g.update(copy.deepcopy(GROUP_DEFAULTS[kind]))
    return g


class Project:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.cfg = copy.deepcopy(DEFAULTS)
        self.groups = []
        self.load()

    # -- files --------------------------------------------------------------
    def path(self, name):
        return os.path.join(self.folder, name)

    def load(self):
        if os.path.exists(self.path("project.json")):
            d = json.load(open(self.path("project.json")))
            self.cfg.update(d.get("settings", {}))
            self.groups = d.get("groups", [])

    def save(self):
        json.dump({"settings": self.cfg, "groups": self.groups},
                  open(self.path("project.json"), "w"), indent=2)

    # -- image ----------------------------------------------------------------
    def set_image(self, src):
        """Copy a picture into the project, resized to the working size.
        Resets the segmentation and groups."""
        im = Image.open(src).convert("RGB")
        s = self.cfg["working_size"] / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(self.path("image.png"))
        for f in ("regions.npy", "sam_masks.npz"):
            if os.path.exists(self.path(f)):
                os.remove(self.path(f))
        shutil.rmtree(self.path("history"), ignore_errors=True)   # undo steps of the old picture
        self.groups = []
        self.save()

    def has_image(self):
        return os.path.exists(self.path("image.png"))

    def image(self):
        """(H, W, 3) float32 in [0, 1]."""
        return np.asarray(Image.open(self.path("image.png")).convert("RGB"),
                          dtype=np.float32) / 255

    # -- geometry -------------------------------------------------------------
    def board(self):
        h, w = self.image().shape[:2]
        bw = float(self.cfg["board_width_cm"])
        return bw, bw * h / w, w / bw          # width, height (cm), pixels per cm

    # -- regions / groups -----------------------------------------------------
    def regions(self):
        return np.load(self.path("regions.npy")) if os.path.exists(self.path("regions.npy")) else None

    def set_regions(self, lab):
        np.save(self.path("regions.npy"), lab.astype(np.int32))

    def group_of(self, region_id):
        for g in self.groups:
            if region_id in g["members"]:
                return g
        return None

    def ordered_groups(self):
        return sorted(self.groups, key=lambda g: g["order"])

    def group_map(self):
        """(H, W) index into ordered_groups() for every pixel (-1 = none)."""
        lab = self.regions()
        lut = np.full(lab.max() + 1, -1, int)
        for i, g in enumerate(self.ordered_groups()):
            for m in g["members"]:
                if m < len(lut):
                    lut[m] = i
        return lut[lab]
