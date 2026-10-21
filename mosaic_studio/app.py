"""Local web app: `python -m mosaic_studio app [picture] [--projects DIR]`.

Tabs follow the steps: 1 picture, 2 regions & groups, 3 styles & build,
4 result & downloads. Every change is saved in the project folder, so closing
the browser loses nothing.

A status line above the tabs always says what is happening: long steps
(finding regions, building) run in a background thread (`Task`) and the page
shows their current stage, a progress bar and the elapsed time, then what to do
next. Edits of regions, groups and styles can be undone and redone
(history.py).
"""
from __future__ import annotations

import csv
import html
import json
import os
import threading
import time
import traceback
import uuid

import gradio as gr
import numpy as np
from PIL import Image

from . import groups as G
from . import pipeline, render, segment, shapes
from .history import History
from .project import STYLE_HELP, STYLES, Project

PROJECTS = os.path.abspath("mosaic_projects")
STRAIGHT = ("brick", "grid", "rows")
MODES = ["Select regions", "Add a missing shape"]

EDITOR_JS = open(os.path.join(os.path.dirname(__file__), "editor.js")).read()
EDITOR_HTML = """
<div class="mz-ed">
  <div class="mz-bar">
    <span class="mz-group">
      <button data-act="draw" class="on" title="Draw (D)">✏️ Draw</button>
      <button data-act="erase" title="Erase segments (E)">✂️ Erase</button>
    </span>
    <span class="mz-group">
      <button data-act="undo" title="Undo (⌘Z)">↶</button>
      <button data-act="redo" title="Redo (⇧⌘Z)">↷</button>
    </span>
    <span class="mz-group">
      <button data-act="zoomout" title="Zoom out (−)">−</button>
      <span class="mz-zoom">100%</span>
      <button data-act="zoomin" title="Zoom in (+)">+</button>
      <button data-act="fit" title="Whole picture (0)">Fit</button>
    </span>
    <span class="mz-group mz-end">
      <button data-act="cancel">Cancel</button>
      <button data-act="confirm" class="mz-primary">✓ Confirm shape</button>
    </span>
  </div>
  <div class="mz-msg"></div>
  <div class="mz-stage"><canvas></canvas></div>
</div>
"""

CSS = """
.status {padding: 12px 16px; border-radius: 10px; font-size: 15px; line-height: 1.5;
         border: 1px solid var(--border-color-primary); background: var(--block-background-fill)}
.status.busy {border-color: var(--color-accent)}
.status.ok {border-color: #3a9d5d}
.status.err {border-color: #d9534f}
.status .bar {height: 8px; border-radius: 4px; margin-top: 8px; overflow: hidden;
              background: var(--border-color-primary)}
.status .bar div {height: 100%; background: var(--color-accent); transition: width .4s}
.hint {opacity: .7; font-size: 13px}
table.groups {width: 100%; border-collapse: collapse; font-size: 14px}
table.groups th {text-align: left; font-weight: 600; opacity: .7}
table.groups td, table.groups th {padding: 3px 6px; border-bottom: 1px solid var(--border-color-primary)}
table.groups tr.focus td {background: var(--color-accent-soft)}
.chip {display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: middle}
.mz-ed {display: flex; flex-direction: column; gap: 6px}
.mz-bar {display: flex; flex-wrap: wrap; gap: 10px; align-items: center}
.mz-group {display: inline-flex; gap: 4px; align-items: center}
.mz-end {margin-left: auto}
.mz-bar button {padding: 5px 11px; border-radius: 7px; cursor: pointer; font-size: 14px;
                border: 1px solid var(--border-color-primary); background: var(--button-secondary-background-fill);
                color: var(--body-text-color)}
.mz-bar button.on {background: var(--color-accent-soft); border-color: var(--color-accent)}
.mz-bar button.mz-primary {background: var(--button-primary-background-fill);
                           color: var(--button-primary-text-color); border-color: transparent}
.mz-bar button:disabled {opacity: .4; cursor: default}
.mz-zoom {min-width: 48px; text-align: center; font-size: 13px; opacity: .8}
.mz-msg {font-size: 14px; min-height: 1.4em}
.mz-stage {height: calc(100vh - 300px); min-height: 420px; border-radius: 8px; overflow: hidden;
           position: relative}
.mz-stage canvas {position: absolute; inset: 0; touch-action: none; display: block}
"""


# --------------------------------------------------------------------------- #
# Long steps in the background
# --------------------------------------------------------------------------- #
class Task:
    """A long step run in a background thread: fn(log, progress) does the work
    and reports progress(fraction, stage). The app polls the task twice a
    second to show the stage and a progress bar instead of a frozen page."""

    def __init__(self, title, fn):
        self.title, self.lines = title, []
        self.frac, self.stage = 0.0, "starting"
        self.result = self.error = None
        self.done = False
        self.t0 = time.time()
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def log(self, msg):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def progress(self, frac, stage):
        self.frac, self.stage = min(max(float(frac), 0.0), 1.0), stage

    def _run(self, fn):
        try:
            self.result = fn(self.log, self.progress)
        except Exception as e:
            self.error = e
            self.log(traceback.format_exc())
        finally:
            self.done = True

    def elapsed(self):
        s = int(time.time() - self.t0)
        return f"{s // 60}:{s % 60:02d}"

    def text(self):
        return "\n".join(self.lines[-80:])


# --------------------------------------------------------------------------- #
# Small helpers (no Gradio components)
# --------------------------------------------------------------------------- #
esc = html.escape


def status_box(kind, text, hint="", frac=None):
    icon = {"idle": "👋", "busy": "⏳", "ok": "✅", "err": "⚠️", "info": "ℹ️"}[kind]
    bar = f"<div class='bar'><div style='width:{100 * frac:.0f}%'></div></div>" if frac is not None else ""
    hint = f"<div class='hint'>{hint}</div>" if hint else ""
    return f"<div class='status {kind}'>{icon} {text}{bar}{hint}</div>"


def busy_box(task, hint=""):
    return status_box("busy", f"<b>{task.title}</b>: {task.stage}… "
                              f"<span class='hint'>({task.elapsed()} so far, {100 * task.frac:.0f}%)</span>",
                      hint, task.frac)


def open_project(name):
    return Project(os.path.join(PROJECTS, name))


def project_names():
    if not os.path.isdir(PROJECTS):
        return []
    return sorted(d for d in os.listdir(PROJECTS)
                  if os.path.exists(os.path.join(PROJECTS, d, "project.json")))


def unique_name(name):
    base, k = name, 2
    while os.path.exists(os.path.join(PROJECTS, name, "project.json")):
        name, k = f"{base}_{k}", k + 1
    return name


def need(pn):
    """The open project, which must have regions (for every editing step)."""
    if not pn:
        raise gr.Error("No project is open yet: start one in 1 · Picture (or open an existing one).")
    p = open_project(pn)
    if p.regions() is None:
        raise gr.Error("This project has no regions yet: use Find regions in 1 · Picture.")
    return p


def regions_text(sel):
    sel = sorted(sel)
    return f"region {sel[0]}" if len(sel) == 1 else "regions " + ", ".join(map(str, sel))


def short(label, n=34):
    return label if len(label) <= n else label[:n - 1] + "…"


def style_label(g):
    return f"{g['style']} {g['angle']:.0f}°" if g["style"] in STRAIGHT else g["style"]


def size_text(wh, bw, tc, gc):
    """Board size and a rough tile count, so the tile size is easy to judge."""
    try:
        w, h = wh
        bw, tc, gc = float(bw), float(tc), float(gc or 0)
        bh = bw * h / w
        n = bw * bh / (tc + gc) ** 2
    except (TypeError, ValueError, ZeroDivisionError):
        return ""
    slow = " (a big one: building will take a few minutes)" if n > 25_000 else ""
    return f"<small>Board {bw:g} × {bh:.1f} cm · roughly {n:,.0f} tiles{slow}</small>"


def selection_text(p, sel):
    if not sel:
        return "_Nothing selected yet: click regions in the picture._"
    by = {}
    for r in sel:
        g = p.group_of(r)
        by.setdefault(g["name"] if g else "no group", []).append(r)
    parts = [f"{', '.join(map(str, rs))} (in *{name}*)" for name, rs in by.items()]
    return f"**{len(sel)} selected:** " + "; ".join(parts)


def groups_table(p, focus=None):
    """Groups front to back, with the colour they are tinted in on the picture."""
    groups = p.ordered_groups()
    cols = render.group_colors(len(groups))
    rows = []
    for i, g in enumerate(groups):
        cls = " class='focus'" if g["name"] == focus else ""
        members = ", ".join(map(str, g["members"]))
        rows.append(f"<tr{cls}><td><span class='chip' style='background:rgb{cols[i]}'></span></td>"
                    f"<td>{i + 1}</td><td><b>{esc(g['name'])}</b></td><td>{g['kind']}</td>"
                    f"<td>{style_label(g)}</td><td>{esc(short(members, 18))}</td></tr>")
    return ("<table class='groups'><tr><th></th><th>#</th><th>group</th><th>kind</th>"
            "<th>fill</th><th>regions</th></tr>" + "".join(rows) + "</table>"
            "<div class='hint'>#1 is at the front: where two groups touch, the one higher in "
            "this list keeps the clean outline.</div>")


# --------------------------------------------------------------------------- #
def build_app(initial_picture=None):
    with gr.Blocks(title="Mosaic Studio", fill_width=True) as demo:
        proj = gr.State(None)          # project name
        selected = gr.State([])        # selected region ids

        gr.Markdown("# Mosaic Studio\n**1 Picture** → **2 Regions** (check the shapes and "
                    "groups) → **3 Styles & build** → **4 Result**. Every step has automatic "
                    "defaults, and everything is saved as you go.")
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                status = gr.HTML(status_box("idle", "Start in <b>1 · Picture</b>: choose a picture, "
                                                    "or open a project you made before."))
            with gr.Column(scale=1, min_width=230):
                undo_btn = gr.Button("↶ Undo", interactive=False, size="sm")
                redo_btn = gr.Button("↷ Redo", interactive=False, size="sm")

        with gr.Tabs() as tabs:
            # ------------------------------------------------------------ tab 1
            with gr.Tab("1 · Picture", id="picture"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Start a new mosaic")
                        picture = gr.Image(type="filepath", label="Picture", value=initial_picture,
                                           height=360)
                        name = gr.Textbox(label="Project name",
                                          placeholder="optional: taken from the picture's file name")
                        with gr.Row():
                            board_w = gr.Number(30, label="Board width (cm)", minimum=1)
                            tile_cm = gr.Number(0.4, label="Tile size (cm)", minimum=0.05)
                            grout_cm = gr.Number(0.04, label="Grout width (cm)", minimum=0)
                        size1 = gr.Markdown("")
                        with gr.Accordion("Advanced", open=False):
                            model = gr.Radio(["vit_l", "vit_b"], value="vit_l",
                                             label="Segmentation model",
                                             info="vit_l finds shapes better; vit_b is about 3× faster")
                            detail = gr.Slider(0.1, 2.0, 0.3, step=0.1,
                                               label="Smallest region (% of the picture)",
                                               info="smaller = more, finer regions")
                        with gr.Row():
                            seg_btn = gr.Button("Find regions →", variant="primary")
                            auto_btn = gr.Button("Do everything automatically")
                        gr.Markdown(
                            "<small>**Find regions** cuts the picture into shapes so you can check "
                            "them in step 2. It takes a few minutes on a laptop; the bar at the top "
                            "shows each stage. **Do everything automatically** goes straight to a "
                            "finished mosaic with the automatic choices (you can still change them "
                            "afterwards).</small>")
                    with gr.Column(scale=2):
                        gr.Markdown("### …or continue a project")
                        existing = gr.Dropdown(project_names(), value=None, label="Your projects")
                        open_btn = gr.Button("Open project")
                        gr.Markdown(f"<small>Projects are saved in `{PROJECTS}`, one folder each. "
                                    "Opening one brings back its regions, groups, styles and "
                                    "mosaic exactly as you left them.</small>")
                with gr.Accordion("Details (log)", open=False):
                    log1 = gr.Textbox(lines=12, show_label=False, autoscroll=True)

            # ------------------------------------------------------------ tab 2
            with gr.Tab("2 · Regions", id="regions"):
                gr.Markdown(
                    "The picture is cut into numbered **regions**. Regions are collected in "
                    "**groups** (names on the picture, one colour tint each). **Each group is "
                    "tiled as one shape**: its rows of tiles follow the group's outline. To keep "
                    "a detail (an eye, a hand) as its own shape, give it its own group.")
                with gr.Row():
                    with gr.Column(scale=5):
                        overlay = gr.Image(label="Regions (numbers) and groups (names)",
                                           interactive=False, height="calc(100vh - 260px)",
                                           format="jpeg")
                        editor = gr.HTML("", html_template=EDITOR_HTML, js_on_load=EDITOR_JS,
                                         apply_default_css=False, visible=False)
                    with gr.Column(scale=2, min_width=340, visible=False) as draw_panel:
                        draw_title = gr.Markdown("")
                        draw_help = gr.Markdown("")
                        with gr.Group(visible=False) as new_opts:
                            draw_name = gr.Textbox(label="Name of the new group",
                                                   placeholder="e.g. hand")
                            draw_kind = gr.Radio(["object", "background"], value="object",
                                                 label="kind")
                    with gr.Column(scale=2, min_width=340) as normal_panel:
                        gr.Markdown("#### ① Select regions")
                        mode = gr.Radio(MODES, value=MODES[0], show_label=False,
                                        info="Add a missing shape: click on something the regions "
                                             "missed and SAM guesses its outline")
                        sel_info = gr.Markdown(selection_text(None, []))
                        clear_btn = gr.Button("Clear selection", size="sm")
                        gr.Markdown("<small>Or outline a shape yourself:</small>")
                        with gr.Row():
                            draw_btn = gr.Button("✏️ Draw new shape", size="sm")
                            modify_btn = gr.Button("✏️ Modify selected shape", size="sm")

                        gr.Markdown("#### ② Put the selection in a group\n<small>The selected "
                                    "regions are **fused** with the group into one shape: the "
                                    "lines between them disappear.</small>")
                        with gr.Row(equal_height=True):
                            add_target = gr.Dropdown([], label="Existing group", scale=2)
                            add_btn = gr.Button("Add to this group", scale=1)
                        with gr.Row(equal_height=True):
                            new_name = gr.Textbox(label="…or a new group named",
                                                  placeholder="e.g. face", scale=2)
                            new_kind = gr.Radio(["object", "background"], value="object",
                                                label="kind", scale=2)
                        new_btn = gr.Button("Create new group", variant="primary")

                        gr.Markdown("#### ③ Groups, front to back")
                        groups_box = gr.HTML("")
                        edit_group = gr.Dropdown([], label="Edit group")
                        with gr.Row():
                            fwd_btn = gr.Button("▲ Bring forward", size="sm")
                            back_btn = gr.Button("▼ Send back", size="sm")
                            pick_btn = gr.Button("Select its regions", size="sm")
                        with gr.Row(equal_height=True):
                            rename_box = gr.Textbox(label="Rename to", scale=2)
                            rename_btn = gr.Button("Rename", scale=1)
                        kind_radio = gr.Radio(["object", "background"], label="Kind",
                                              info="objects start with rows echoing their "
                                                   "outline, backgrounds with brick rows")
                        with gr.Accordion("Start over", open=False):
                            reset_detail = gr.Slider(0.1, 2.0, 0.3, step=0.1,
                                                     label="Smallest region (% of the picture)")
                            reset_btn = gr.Button("Recompute regions & groups automatically")
                            gr.Markdown("<small>Replaces all your region and group edits "
                                        "(Undo brings them back). Uses the shapes already "
                                        "found, so it is quick.</small>")
                with gr.Accordion("Details (log)", open=False):
                    log2 = gr.Textbox(lines=8, show_label=False, autoscroll=True)

            # ------------------------------------------------------------ tab 3
            with gr.Tab("3 · Styles & build", id="styles"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=340):
                        gr.Markdown("#### How each group is filled\n<small>Changes are saved as "
                                    "you make them; press **Build mosaic** to see them.</small>")
                        sgroup = gr.Dropdown([], label="Group")
                        style = gr.Dropdown(STYLES, label="Fill style")
                        style_help = gr.Markdown("")
                        angle = gr.Slider(-90, 90, 0, step=1, label="Row angle (°)")
                        outline_rows = gr.Slider(0, 6, 1, step=1, label="Rows following the outline")
                        inner = gr.Slider(0, 1, 1, step=0.05, label="Inner lines followed",
                                          info="0 = outline only, 1 = the border of every region "
                                               "inside the group (only for groups not fused yet)")
                        ncol = gr.Slider(1, 24, 8, step=1, label="Colours in this group")
                        fanx = gr.Slider(0, 1, 0.5, step=0.01, label="Fan centre: x (0 = left)")
                        fany = gr.Slider(0, 1, 0.0, step=0.01, label="Fan centre: y (0 = bottom)")
                        groups_box3 = gr.HTML("")

                        gr.Markdown("#### Whole mosaic")
                        with gr.Row():
                            board3 = gr.Number(30, label="Board width (cm)", minimum=1)
                            tile3 = gr.Number(0.4, label="Tile size (cm)", minimum=0.05)
                            grout3 = gr.Number(0.04, label="Grout width (cm)", minimum=0)
                        size3 = gr.Markdown("")
                        saturation = gr.Slider(0.8, 1.6, 1.15, step=0.05, label="Saturation")
                        contrast = gr.Slider(0, 30, 12, step=1, label="Outline contrast between groups")
                        fill_gaps = gr.Checkbox(True, label="Fill leftover gaps with tiles")
                        snug = gr.Slider(0, 8, 0, step=1, label="Snugging rounds",
                                         info="tightens the joints; each round adds time")
                        build_btn = gr.Button("Build mosaic →", variant="primary")
                    with gr.Column(scale=2):
                        lines_img = gr.Image(label="Guide lines of the last build", interactive=False)
                with gr.Accordion("Details (log)", open=False):
                    log3 = gr.Textbox(lines=12, show_label=False, autoscroll=True)

            # ------------------------------------------------------------ tab 4
            with gr.Tab("4 · Result", id="result"):
                result_info = gr.Markdown("")
                mosaic_img = gr.Image(label="Mosaic", interactive=False)
                files = gr.File(label="Downloads (mosaic.png, tiles.csv, shopping_list.csv)",
                                file_count="multiple")
                shopping = gr.Dataframe(headers=["colour", "r", "g", "b", "tiles", "used in"],
                                        label="Shopping list (tiles per colour)", interactive=False)

        # ================================================================ views
        LONG = [seg_btn, auto_btn, open_btn, reset_btn, build_btn]
        STYLE_CTRLS = [style, style_help, angle, outline_rows, inner, ncol, fanx, fany]
        VIEW = [overlay, sel_info, add_target, edit_group, kind_radio, sgroup, groups_box,
                groups_box3, undo_btn, redo_btn, *STYLE_CTRLS]
        SETTINGS = [board3, tile3, grout3, size3, saturation, contrast, fill_gaps, snug, reset_detail]
        RESULTS = [lines_img, mosaic_img, files, shopping, result_info]
        PROJECT = [proj, selected, *VIEW, *SETTINGS, *RESULTS]
        RUN = [status, tabs, existing, *LONG, *PROJECT]

        def buttons(on):
            return {b: gr.update(interactive=on) for b in LONG}

        def history_view(p):
            u, r = History(p).next_labels()
            return {undo_btn: gr.update(value=f"↶ Undo: {short(u)}" if u else "↶ Undo",
                                        interactive=bool(u)),
                    redo_btn: gr.update(value=f"↷ Redo: {short(r)}" if r else "↷ Redo",
                                        interactive=bool(r))}

        def style_view(g):
            if not g:
                return {}
            st = g["style"]
            return {style: st, style_help: f"<small>{STYLE_HELP[st]}</small>",
                    angle: gr.update(value=g["angle"], visible=st in STRAIGHT),
                    outline_rows: g["outline_rows"],
                    inner: gr.update(value=g["inner_detail"], visible=st == "echo"),
                    ncol: g["colors"],
                    fanx: gr.update(value=g["fan_center"][0], visible=st == "fan"),
                    fany: gr.update(value=g["fan_center"][1], visible=st == "fan")}

        def view(pn, sel, focus=None):
            """Everything that shows the regions and groups (tabs 2 and 3, undo/redo)."""
            p = open_project(pn)
            names = [g["name"] for g in p.ordered_groups()]
            pick = focus if focus in names else (names[0] if names else None)
            g = G.by_name(p, pick)
            return {overlay: (pipeline.preview_regions(p, sel, save=not sel)
                              if p.regions() is not None else None),
                    sel_info: selection_text(p, sel),
                    add_target: gr.update(choices=names, value=pick),
                    edit_group: gr.update(choices=names, value=pick),
                    kind_radio: g["kind"] if g else None,
                    sgroup: gr.update(choices=names, value=pick),
                    groups_box: groups_table(p, pick), groups_box3: groups_table(p, pick),
                    **history_view(p), **style_view(g)}

        def settings_view(p):
            c = p.cfg
            wh = Image.open(p.path("image.png")).size if p.has_image() else None
            return {board3: c["board_width_cm"], tile3: c["tile_cm"], grout3: c["grout_cm"],
                    size3: size_text(wh, c["board_width_cm"], c["tile_cm"], c["grout_cm"]),
                    saturation: c["saturation"], contrast: c["contrast"],
                    fill_gaps: c["fill_gaps"], snug: c["snug_rounds"],
                    reset_detail: c["min_region_pct"]}

        def results_view(p):
            if not pipeline.is_built(p):
                return {lines_img: None, mosaic_img: None, files: None, shopping: None,
                        result_info: "_No mosaic yet: build it in **3 · Styles & build**._"}
            rows = list(csv.reader(open(p.path("shopping_list.csv"))))[1:]
            n = sum(1 for _ in open(p.path("tiles.csv"))) - 1
            lines = p.path("preview_2_lines.png")
            return {lines_img: lines if os.path.exists(lines) else None,
                    mosaic_img: p.path("mosaic.png"),
                    files: [p.path(f) for f in pipeline.OUTPUTS],
                    shopping: rows,
                    result_info: f"**{n:,} tiles in {len(rows)} colours.** The files are also "
                                 f"in `{os.path.abspath(p.folder)}`. `tiles.csv` gives every "
                                 "tile's centre in cm from the bottom-left corner, its angle and "
                                 "colour."}

        def project_view(pn, sel=(), focus=None):
            p = open_project(pn)
            return {proj: pn, selected: list(sel), **view(pn, list(sel), focus),
                    **settings_view(p), **results_view(p)}

        def follow(task, log_box, hint=""):
            """Show a running task in the status line until it ends; returns it."""
            while not task.done:
                yield {status: busy_box(task, hint), log_box: task.text()}
                time.sleep(0.5)
            return task

        def failed(task, log_box):
            msg = esc(f"{type(task.error).__name__}: {task.error}")
            return {status: status_box("err", f"<b>{task.title}</b> stopped with an error: {msg}",
                                       "The full error is in Details (log) below."),
                    log_box: task.text()}

        # ================================================================ tab 1
        SEG_HINT = ("Nothing to click until it is done. A new picture takes a few minutes on a "
                    "laptop (the first time ever also downloads the model).")

        def start_project(pic, nm, bw, tc, gc, mdl, det):
            if not pic:
                raise gr.Error("Choose a picture first: drop it in the Picture box.")
            base = (nm or os.path.splitext(os.path.basename(pic))[0]).strip()
            pn = unique_name(base)
            p = open_project(pn)
            p.cfg.update(board_width_cm=float(bw), tile_cm=float(tc), grout_cm=float(gc),
                         sam_model=mdl, min_region_pct=float(det))
            p.set_image(pic)
            note = (f" It is called <b>{esc(pn)}</b> because a project <b>{esc(base)}</b> "
                    "already exists." if pn != base else "")
            return pn, p, note

        def segmented_text(p):
            return f"{len(np.unique(p.regions()))} regions in {len(p.groups)} groups"

        def new_mosaic(pic, nm, bw, tc, gc, mdl, det):
            pn, p, note = start_project(pic, nm, bw, tc, gc, mdl, det)
            yield {**buttons(False), log1: ""}
            task = yield from follow(Task("Finding regions", lambda log, prog:
                                          segment.segment_project(p, log, progress=prog)),
                                     log1, SEG_HINT)
            if task.error:
                yield {**failed(task, log1), **buttons(True)}
                return
            yield {status: status_box("ok", f"<b>Regions found</b> in {task.elapsed()}: "
                                            f"{segmented_text(p)}.{note}",
                                      "Next: check the regions and groups in <b>2 · Regions</b> "
                                      "(opened for you), then choose styles in <b>3</b>."),
                   log1: task.text(), tabs: gr.Tabs(selected="regions"),
                   existing: gr.update(choices=project_names(), value=pn),
                   **buttons(True), **project_view(pn)}

        def auto_mosaic(pic, nm, bw, tc, gc, mdl, det):
            pn, p, note = start_project(pic, nm, bw, tc, gc, mdl, det)
            yield {**buttons(False), log1: ""}
            task = yield from follow(Task("Making the mosaic", lambda log, prog:
                                          pipeline.run_all(p, None, log, prog)),
                                     log1, SEG_HINT)
            if task.error:
                yield {**failed(task, log1), **buttons(True)}
                return
            yield {status: status_box("ok", f"<b>Mosaic done</b> in {task.elapsed()}: "
                                            f"{len(task.result[0]):,} tiles.{note}",
                                      "It is in <b>4 · Result</b> (opened for you). To change "
                                      "it, edit the groups in <b>2</b> or the styles in <b>3</b> "
                                      "and build again."),
                   log1: task.text(), tabs: gr.Tabs(selected="result"),
                   existing: gr.update(choices=project_names(), value=pn),
                   **buttons(True), **project_view(pn)}

        def open_existing(pn):
            if not pn:
                raise gr.Error("Choose one of your projects in the list first.")
            p = open_project(pn)
            if not p.has_image():
                raise gr.Error("This project has no picture: start a new mosaic instead.")
            log_text = ""
            if p.regions() is None or not p.groups:
                yield {**buttons(False), log1: ""}
                task = yield from follow(Task("Finding regions", lambda log, prog:
                                              segment.segment_project(p, log, progress=prog)),
                                         log1, SEG_HINT)
                if task.error:
                    yield {**failed(task, log1), **buttons(True)}
                    return
                log_text = task.text()
            hint = ("It already has a mosaic: see <b>4 · Result</b>. Edit the groups here or "
                    "the styles in <b>3</b>, then build again." if pipeline.is_built(p) else
                    "Next: check the regions and groups here, then choose styles in <b>3</b>.")
            yield {status: status_box("ok", f"Opened <b>{esc(pn)}</b>: {segmented_text(p)}.", hint),
                   log1: log_text, tabs: gr.Tabs(selected="regions"), **buttons(True),
                   **project_view(pn)}

        seg_btn.click(new_mosaic, [picture, name, board_w, tile_cm, grout_cm, model, detail],
                      [*RUN, log1], show_progress="hidden")
        auto_btn.click(auto_mosaic, [picture, name, board_w, tile_cm, grout_cm, model, detail],
                       [*RUN, log1], show_progress="hidden")
        open_btn.click(open_existing, [existing], [*RUN, log1], show_progress="hidden")

        def picture_size(pic, bw, tc, gc):
            try:
                return size_text(Image.open(pic).size, bw, tc, gc) if pic else ""
            except OSError:
                return ""
        for c in (picture, board_w, tile_cm, grout_cm):
            c.change(picture_size, [picture, board_w, tile_cm, grout_cm], [size1],
                     show_progress="hidden")
        demo.load(lambda: gr.update(choices=project_names()), None, [existing])

        # ================================================================ tab 2
        def click_region(pn, sel, m, evt: gr.SelectData):
            p = need(pn)
            x, y = int(evt.index[0]), int(evt.index[1])
            if m == MODES[0]:
                r = int(p.regions()[y, x])
                sel = [s for s in sel if s != r] if r in sel else sel + [r]
                yield {selected: sel, sel_info: selection_text(p, sel),
                       overlay: pipeline.preview_regions(p, sel, save=False)}
                return
            task = yield from follow(Task("Adding a shape", lambda log, prog:
                                          segment.mask_from_click(p, x, y, log, prog)), log2,
                                     "The first click on a picture takes longer (up to a "
                                     "minute); the next ones are quick.")
            if task.error:
                yield failed(task, log2)
                return
            mask = task.result
            if not mask.any():
                yield {status: status_box("err", "No shape found there.",
                                          "Try clicking nearer the middle of the shape.")}
                return
            new = History(p).apply(f"add a shape at ({x}, {y})",
                                   lambda: segment.add_region_from_mask(p, mask))
            g = p.group_of(new)
            yield {selected: [new], mode: MODES[0], log2: task.text(),
                   status: status_box("ok", f"New region <b>{new}</b> added (for now part of "
                                            f"<b>{esc(g['name']) if g else 'no group'}</b>).",
                                      "It is selected: add it to a group or make it a new group "
                                      "(②). Click mode is back to Select regions."),
                   **view(pn, [new], g["name"] if g else None)}

        overlay.select(click_region, [proj, selected, mode],
                       [selected, mode, status, log2, *VIEW], show_progress="hidden")

        # -------------------------------------------------------- shape editor
        EDITOR_UI = [editor, overlay, normal_panel, draw_panel, draw_title, draw_help, new_opts,
                     undo_btn, redo_btn]
        NEW_HELP = ("**Drag** to draw the outline. Stop whenever you like: the line keeps a "
                    "**blue dot** at each open end. **Drag from a dot** to carry on, and end on a "
                    "dot to connect two lines or close the shape. **✂️ Erase** removes the "
                    "segment you click (drag to erase several).\n\n"
                    "**Zoom:** pinch, ⌘/Ctrl + scroll, or + −. **Move around:** scroll, or "
                    "hold Space (or the right button) and drag. **⌘Z** undoes a stroke.\n\n"
                    "The new shape becomes a group of its own, at the front. It takes its area "
                    "from whatever was there.")
        EDIT_HELP = ("The outline is cut into short **segments** (between the white dots). "
                     "**✂️ Erase** the segments that are wrong (click, or drag along them), then "
                     "**✏️ Draw** the missing part from one **blue dot** to the other.\n\n"
                     "**Zoom:** pinch, ⌘/Ctrl + scroll, or + −. **Move around:** scroll, or "
                     "hold Space (or the right button) and drag. **⌘Z** undoes a step.\n\n"
                     "Area the shape gains is taken from its neighbours; area it loses goes "
                     "to the nearest neighbour.")

        def editor_session(p, mode, region=None):
            lab = p.regions()
            H, W = lab.shape
            spacing = max(8, round(max(W, H) / 60))     # length of an erasable segment, px
            # outline points closer than the editor's own point step (spacing / 3)
            loops = shapes.region_outline(lab, region, step=spacing / 3.2) if region else []
            return json.dumps({"token": uuid.uuid4().hex, "mode": mode, "region": region,
                               "img": shapes.image_data_url(p), "w": W, "h": H,
                               "loops": loops, "spacing": spacing})

        def open_editor(pn, sel, mode):
            p = need(pn)
            if mode == "edit":
                if not sel:
                    raise gr.Error("Select the region to modify first: click it in the picture, "
                                   "then press Modify selected shape.")
                r = int(sel[-1])                    # several selected: the one clicked last
                g = p.group_of(r)
                session = editor_session(p, "edit", r)
                title = f"### Modifying region {r}" + (f" (in *{g['name']}*)" if g else "")
                what = f"Modifying region <b>{r}</b>" + (
                    " (the last one you clicked)" if len(sel) > 1 else "")
            else:
                session = editor_session(p, "new")
                title, what = "### Drawing a new shape", "Drawing a new shape"
            return {editor: gr.update(value=session, visible=True), overlay: gr.update(visible=False),
                    normal_panel: gr.update(visible=False), draw_panel: gr.update(visible=True),
                    draw_title: title, draw_help: EDIT_HELP if mode == "edit" else NEW_HELP,
                    new_opts: gr.update(visible=mode == "new"),
                    undo_btn: gr.update(interactive=False), redo_btn: gr.update(interactive=False),
                    status: status_box("info", f"{what}: confirm or cancel above the picture when "
                                               "you are done.")}
        draw_btn.click(lambda pn, sel: open_editor(pn, sel, "new"), [proj, selected],
                       [status, *EDITOR_UI])
        modify_btn.click(lambda pn, sel: open_editor(pn, sel, "edit"), [proj, selected],
                         [status, *EDITOR_UI])

        def editor_done(pn, sel, value, name, kind):
            out = _editor_done(pn, sel, value, name, kind)
            if overlay in out and not isinstance(out[overlay], dict):     # a new picture: show it too
                out[overlay] = gr.update(value=out[overlay], visible=True)
            return out

        def _editor_done(pn, sel, value, name, kind):
            try:
                res = json.loads(value or "{}")
            except ValueError:
                res = {}
            if not res.get("result"):
                return {}
            p = need(pn)
            close = {editor: gr.update(value="", visible=False), overlay: gr.update(visible=True),
                     normal_panel: gr.update(visible=True), draw_panel: gr.update(visible=False)}
            if res.get("action") != "confirm":
                return {**close, selected: [], **view(pn, []),
                        status: status_box("info", "Closed the drawing: nothing changed.")}
            lab = p.regions()
            if res["mode"] == "new":
                mask = shapes.rasterize(res["loops"], lab.shape)
                if mask.sum() < 20:
                    raise gr.Error("That shape is too small: draw a bigger outline.")
                nm = (name or "").strip() or "new shape"
                made = History(p).apply(f"draw new shape '{nm}'",
                                        lambda: shapes.add_shape(p, mask, nm, kind))
                return {**close, selected: [], draw_name: "", **view(pn, [], made),
                        status: status_box("ok", f"Added your shape as the group <b>{esc(made)}</b> "
                                                 "(#1, at the front).",
                                           "Choose its fill style in <b>3 · Styles & build</b>. "
                                           "Not right? Press Undo, or select it and Modify.")}
            r = int(res["region"])
            if not (lab == r).any():
                raise gr.Error(f"Region {r} no longer exists: cancel and start again.")
            mask = shapes.edited_mask(lab, r, res.get("before", []), res["loops"])
            if mask.sum() < 20:
                raise gr.Error("The shape would be (almost) empty: draw it bigger, or cancel.")
            if np.array_equal(mask, lab == r):
                return {**close, **view(pn, [], (p.group_of(r) or {}).get("name")),
                        selected: [], status: status_box("info", "The outline did not change.")}
            History(p).apply(f"reshape region {r}", lambda: shapes.reshape_region(p, r, mask))
            g = p.group_of(r)
            return {**close, selected: [], **view(pn, [], g["name"] if g else None),
                    status: status_box("ok", f"Region <b>{r}</b> reshaped.",
                                       "Press Undo to get the old outline back. To modify another "
                                       "shape, click it and press Modify selected shape.")}
        editor.submit(editor_done, [proj, selected, editor, draw_name, draw_kind],
                      [selected, draw_name, status, *dict.fromkeys(EDITOR_UI + VIEW)],
                      show_progress="hidden")

        def clear_selection(pn):
            return {selected: [], **view(pn, [])} if pn else {}
        clear_btn.click(clear_selection, [proj], [selected, *VIEW])

        def add_to(pn, sel, grp):
            p = need(pn)
            if not sel:
                raise gr.Error("Select regions first: click them in the picture.")
            if not grp:
                raise gr.Error("Choose the group to add them to.")
            History(p).apply(f"add {regions_text(sel)} to '{grp}'",
                             lambda: G.add_to_group(p, sel, grp))
            rid = G.by_name(p, grp)["members"][0]
            return {selected: [], **view(pn, [], grp),
                    status: status_box("ok", f"Added {regions_text(sel)} to <b>{esc(grp)}</b>. "
                                             f"The group is now one shape (region {rid}).",
                                       "Not what you wanted? Press Undo.")}
        add_btn.click(add_to, [proj, selected, add_target], [selected, status, *VIEW])

        def new_from(pn, sel, nm, kd):
            p = need(pn)
            if not sel:
                raise gr.Error("Select regions first: click them in the picture.")
            nm = (nm or "").strip() or "new group"
            made = History(p).apply(f"new group '{nm}' from {regions_text(sel)}",
                                    lambda: G.new_group_from(p, sel, nm, kd))
            return {selected: [], new_name: "", **view(pn, [], made),
                    status: status_box("ok", f"Created the {kd} <b>{esc(made)}</b> from "
                                             f"{regions_text(sel)}, fused into one shape. It is "
                                             "at the front (#1).",
                                       "Use ▲ / ▼ in ③ to change its place; set its fill "
                                       "style in <b>3 · Styles & build</b>.")}
        new_btn.click(new_from, [proj, selected, new_name, new_kind],
                      [selected, new_name, status, *VIEW])

        def move(pn, sel, grp, d):
            p = need(pn)
            if not grp:
                raise gr.Error("Choose a group in Edit group first.")
            History(p).apply(f"move '{grp}' {'forward' if d < 0 else 'back'}",
                             lambda: G.move(p, grp, d))
            k = G.by_name(p, grp)["order"] + 1
            where = " (the front)" if k == 1 else " (the back)" if k == len(p.groups) else ""
            return {**view(pn, sel, grp),
                    status: status_box("ok", f"<b>{esc(grp)}</b> is now #{k} of "
                                             f"{len(p.groups)}{where}.")}
        fwd_btn.click(lambda a, b, c: move(a, b, c, -1), [proj, selected, edit_group],
                      [status, *VIEW])
        back_btn.click(lambda a, b, c: move(a, b, c, +1), [proj, selected, edit_group],
                       [status, *VIEW])

        def pick_group(pn, grp):
            p = need(pn)
            g = G.by_name(p, grp)
            if not g:
                raise gr.Error("Choose a group in Edit group first.")
            return {selected: list(g["members"]), **view(pn, g["members"], grp)}
        pick_btn.click(pick_group, [proj, edit_group], [selected, *VIEW])

        def rename(pn, sel, grp, new):
            p = need(pn)
            new = (new or "").strip()
            if not grp or not new:
                raise gr.Error("Choose a group and type its new name.")
            if G.by_name(p, new):
                raise gr.Error(f"There is already a group called '{new}'.")
            History(p).apply(f"rename '{grp}' to '{new}'", lambda: G.rename(p, grp, new))
            return {rename_box: "", **view(pn, sel, new),
                    status: status_box("ok", f"Renamed <b>{esc(grp)}</b> to <b>{esc(new)}</b>.")}
        rename_btn.click(rename, [proj, selected, edit_group, rename_box],
                         [rename_box, status, *VIEW])

        def set_kind(pn, sel, grp, kd):
            p = need(pn)
            if not grp or not kd:
                return {}
            History(p).apply(f"make '{grp}' {'an' if kd == 'object' else 'a'} {kd}",
                             lambda: G.set_kind(p, grp, kd))
            g = G.by_name(p, grp)
            return {**view(pn, sel, grp),
                    status: status_box("ok", f"<b>{esc(grp)}</b> is now {'an' if kd == 'object' else 'a'} "
                                             f"{kd}; its fill style was reset to <b>{style_label(g)}</b>, "
                                             f"the usual one for {kd}s.")}
        kind_radio.input(set_kind, [proj, selected, edit_group, kind_radio], [status, *VIEW])

        def focus_group(pn, grp):
            if not pn:
                return {}
            p = open_project(pn)
            g = G.by_name(p, grp)
            return {kind_radio: g["kind"] if g else None, groups_box: groups_table(p, grp)}
        edit_group.input(focus_group, [proj, edit_group], [kind_radio, groups_box])

        def undo(pn, redo=False):
            p = need(pn)
            h = History(p)
            label = h.redo() if redo else h.undo()
            if label is None:
                return {status: status_box("info", f"Nothing to {'redo' if redo else 'undo'}.")}
            return {selected: [], **view(pn, []),
                    status: status_box("ok", f"{'Redone' if redo else 'Undone'}: {esc(label)}.",
                                       "Build the mosaic again in <b>3</b> to see the change "
                                       "there.")}
        undo_btn.click(undo, [proj], [selected, status, *VIEW])
        redo_btn.click(lambda pn: undo(pn, redo=True), [proj], [selected, status, *VIEW])

        def start_over(pn, det):
            p = need(pn)
            p.cfg["min_region_pct"] = float(det)
            p.save()
            yield {**buttons(False), log2: ""}
            task = yield from follow(Task("Recomputing regions", lambda log, prog: History(p).apply(
                "recompute regions & groups",
                lambda: segment.segment_project(p, log, reuse_masks=True, progress=prog))), log2)
            if task.error:
                yield {**failed(task, log2), **buttons(True)}
                return
            yield {status: status_box("ok", f"Regions recomputed: {segmented_text(p)}.",
                                      "Your earlier edits are gone; Undo brings them back."),
                   log2: task.text(), selected: [], **buttons(True), **view(pn, [])}
        reset_btn.click(start_over, [proj, reset_detail],
                        [status, log2, selected, *LONG, *VIEW], show_progress="hidden")

        # ================================================================ tab 3
        def show_group(pn, grp):
            if not pn:
                return {}
            p = open_project(pn)
            return {groups_box3: groups_table(p, grp), **style_view(G.by_name(p, grp))}
        sgroup.input(show_group, [proj, sgroup], [groups_box3, *STYLE_CTRLS])

        def save_style(pn, grp, st, an, orow, inn, nc, fx, fy):
            if not pn or not grp:
                return {}
            p = open_project(pn)
            if not G.by_name(p, grp):
                return {}
            History(p).apply(f"style of '{grp}'", lambda: G.set_style(
                p, grp, style=st, angle=float(an), outline_rows=int(orow), inner_detail=float(inn),
                colors=int(nc), fan_center=[float(fx), float(fy)]))
            g = G.by_name(p, grp)
            return {status: status_box("ok", f"Saved: <b>{esc(grp)}</b> is filled with "
                                             f"<b>{style_label(g)}</b>, {g['outline_rows']} outline "
                                             f"row(s), {g['colors']} colours.",
                                       "Press <b>Build mosaic</b> to see it."),
                    groups_box: groups_table(p, grp), groups_box3: groups_table(p, grp),
                    **history_view(p), **style_view(g)}
        style_inputs = [proj, sgroup, style, angle, outline_rows, inner, ncol, fanx, fany]
        style_outputs = [status, groups_box, groups_box3, undo_btn, redo_btn, *STYLE_CTRLS]
        style.input(save_style, style_inputs, style_outputs, show_progress="hidden")
        for c in (angle, outline_rows, inner, ncol, fanx, fany):
            c.release(save_style, style_inputs, style_outputs, show_progress="hidden")

        def board_size(pn, bw, tc, gc):
            if not pn:
                return ""
            p = open_project(pn)
            return size_text(Image.open(p.path("image.png")).size, bw, tc, gc) if p.has_image() else ""
        for c in (board3, tile3, grout3):
            c.change(board_size, [proj, board3, tile3, grout3], [size3], show_progress="hidden")

        def build(pn, bw, tc, gc, sat, con, fg, sn):
            p = need(pn)
            if not (bw and bw > 0 and tc and tc > 0 and gc is not None and gc >= 0):
                raise gr.Error("Board width and tile size must be above 0 (grout 0 or more).")
            p.cfg.update(board_width_cm=float(bw), tile_cm=float(tc), grout_cm=float(gc),
                         saturation=float(sat), contrast=float(con), fill_gaps=bool(fg),
                         snug_rounds=int(sn))
            p.save()
            yield {**buttons(False), log3: ""}
            task = yield from follow(Task("Building the mosaic", lambda log, prog:
                                          pipeline.build(p, log, prog)), log3,
                                     "About a minute for ~6,000 tiles; more with small tiles "
                                     "or snugging.")
            if task.error:
                yield {**failed(task, log3), **buttons(True)}
                return
            yield {status: status_box("ok", f"<b>Mosaic built</b> in {task.elapsed()}: "
                                            f"{len(task.result[0]):,} tiles.",
                                      "It is in <b>4 · Result</b> (opened for you), with the "
                                      "downloads and the shopping list."),
                   log3: task.text(), tabs: gr.Tabs(selected="result"), **buttons(True),
                   **results_view(p)}
        build_btn.click(build, [proj, board3, tile3, grout3, saturation, contrast, fill_gaps, snug],
                        [status, tabs, log3, *LONG, *RESULTS], show_progress="hidden")
    return demo


def main(picture=None, projects=None, port=7860, share=False):
    global PROJECTS
    if projects:
        PROJECTS = os.path.abspath(projects)
    os.makedirs(PROJECTS, exist_ok=True)
    build_app(picture).launch(server_port=port, share=share, inbrowser=True, css=CSS,
                              allowed_paths=[PROJECTS])      # results may live outside the cwd
