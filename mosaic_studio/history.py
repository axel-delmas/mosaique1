"""Undo / redo for edits of regions and groups (merging, new groups, added
shapes, order, names, kinds, styles).

Every edit is wrapped in `History(project).apply(label, fn)`: the state before
it (the region label map + the groups) is saved as one compressed snapshot in
<project>/history/, so undo survives closing the app. An edit that changes
nothing leaves no step. A new edit clears the redo steps, like in any editor.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid

import numpy as np

MAX_STEPS = 40


class History:
    def __init__(self, project):
        self.p = project
        self.dir = project.path("history")
        self.index = os.path.join(self.dir, "index.json")
        self.undo_steps, self.redo_steps = [], []    # [{"file", "label"}], newest last
        if os.path.exists(self.index):
            d = json.load(open(self.index))
            self.undo_steps, self.redo_steps = d["undo"], d["redo"]

    # -- state ----------------------------------------------------------------
    def _state(self):
        lab = self.p.regions()
        return (lab if lab is not None else np.zeros((0, 0), np.int32)), json.dumps(self.p.groups)

    def _write(self, state, label):
        os.makedirs(self.dir, exist_ok=True)
        name = uuid.uuid4().hex[:12] + ".npz"
        np.savez_compressed(os.path.join(self.dir, name), regions=state[0], groups=state[1])
        return {"file": name, "label": label}

    def _restore(self, step):
        d = np.load(os.path.join(self.dir, step["file"]))
        if d["regions"].size:
            self.p.set_regions(d["regions"])
        self.p.groups = json.loads(str(d["groups"]))
        self.p.save()

    def _drop(self, steps):
        for s in steps:
            f = os.path.join(self.dir, s["file"])
            if os.path.exists(f):
                os.remove(f)

    def _save(self):
        os.makedirs(self.dir, exist_ok=True)
        json.dump({"undo": self.undo_steps, "redo": self.redo_steps}, open(self.index, "w"), indent=1)

    # -- public ---------------------------------------------------------------
    def apply(self, label, fn):
        """Run fn() (an edit of the project) as one undoable step; returns its result."""
        before = self._state()
        out = fn()
        after = self._state()
        if before[1] != after[1] or not np.array_equal(before[0], after[0]):
            self.undo_steps.append(self._write(before, label))
            self._drop(self.redo_steps)
            self.redo_steps = []
            if len(self.undo_steps) > MAX_STEPS:
                self._drop(self.undo_steps[:-MAX_STEPS])
                self.undo_steps = self.undo_steps[-MAX_STEPS:]
            self._save()
        return out

    def undo(self):
        """Go back one step; returns its label, or None if there is nothing to undo."""
        return self._move(self.undo_steps, self.redo_steps)

    def redo(self):
        return self._move(self.redo_steps, self.undo_steps)

    def _move(self, src, dst):
        if not src:
            return None
        step = src.pop()
        dst.append(self._write(self._state(), step["label"]))
        self._restore(step)
        self._drop([step])
        self._save()
        return step["label"]

    def next_labels(self):
        """(label of the step undo would revert, label redo would re-apply), or None."""
        return (self.undo_steps[-1]["label"] if self.undo_steps else None,
                self.redo_steps[-1]["label"] if self.redo_steps else None)

    def clear(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.undo_steps, self.redo_steps = [], []
