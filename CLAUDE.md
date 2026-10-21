# CLAUDE.md — Mosaic Studio (`mosaique/`)

Turns a picture into a **physical mosaic plan**: where each square tile goes (cm), at what
angle, in what colour, plus a shopping list of tiles per colour. Personal project; not a git
repo, no tests, no CI.

The folder holds two things:

1. **`mosaic_studio/`**: the actual product, a Python package with a Gradio web app and a CLI.
   **New work goes here.**
2. **Root-level `*.py` scripts**: an earlier *research phase* (Sept 23–25, 2026) exploring
   differentiable losses, edge fields, deformed grids, a CNN, and guide-line placement. They
   produced the PNGs in `colors/`, `lines/`, `rows/`, `edge_pipeline/`, `small_tiles/`,
   `new_degas/`, `network/`. The good ideas were copied into `mosaic_studio/` (see the
   lineage table below). Treat the scripts as reference and archive.

---

## Running

```bash
source .venv/bin/activate              # Python 3.14 venv, already set up
pip install -r requirements.txt        # numpy scipy scikit-image pillow opencv-python-headless torch segment-anything gradio>=5

python -m mosaic_studio app [picture] [--projects mosaic_projects] [--port 7860]
python -m mosaic_studio run photo.jpg [--name N] [--board-width 30] [--tile 0.4] [--grout 0.04] [--model vit_l|vit_b]
```

- `Mosaic Studio.command` (repo root) is a double-click launcher for macOS: it changes to its
  own folder, activates `.venv` and runs `app`.
- **Run from the repo root.** The projects dir defaults to the relative path `mosaic_projects`.
- SAM weights are cached in `~/.cache/mosaic_studio/sam_vit_{b,l}.pth`. Both are already
  downloaded on this machine. `segment.WEIGHTS` lists the official URL plus an ultralytics mirror.
- Device order is cuda → mps → cpu. `segment._run` falls back to CPU if the GPU fails, and
  `_float32_points` patches SAM so it works on Apple MPS, which has no float64.
- Timing: SAM takes minutes on CPU the first time. Its masks are cached per project in
  `sam_masks.npz`. A rebuild takes about 1 min for ~6k tiles.
- **matplotlib is NOT installed** in the venv and is not in `requirements.txt`. The package
  doesn't need it, but almost every root-level script does (`pip install matplotlib` first).

---

## Pipeline (`mosaic_studio/`)

```
picture ─► segment.py ─► groups.py (user edits) ─► layout.py + geometry.py ─► colors.py ─► render.py / exports
           SAM masks →     merge / new / order /     guide lines per group,    palette per    mosaic.png, tiles.csv,
           regions.npy +   rename / kind             tiles placed greedily     group, outline shopping_list.csv,
           auto groups                                                         contrast       tiles.json
```

| Module | Role |
|---|---|
| `__main__.py` | argparse CLI: `app` / `run` |
| `app.py` | Gradio Blocks UI with 4 tabs (ids `picture`, `regions`, `styles`, `result`), plus a status line (`gr.HTML`) and Undo/Redo above the tabs. Each callback re-opens the `Project` from disk by name (`gr.State` holds only the project name and selected region ids). Handlers return **dicts** `{component: value}`, so output lists can be supersets (`VIEW`, `SETTINGS`, `RESULTS`, `PROJECT`, `RUN`). Long steps (find regions, build, auto, start over, add shape) are **generators**: the work runs in a `Task` thread reporting `progress(frac, stage)`, `follow()` yields the status every 0.5 s, and the final yield switches tab. Every region/group/style edit goes through `History(p).apply(label, fn)` |
| `shapes.py` + `editor.js` | Hand-drawn shapes. `editor.js` is a canvas editor run by a `gr.HTML` (`html_template=EDITOR_HTML`, `js_on_load=EDITOR_JS` in app.py): all drawing, zoom/pan and stroke undo happen in the browser. Python sends a session as the component's value (JSON: token, mode new|edit, region, picture as a JPEG data URL, `loops`, `spacing`); on Confirm/Cancel the JS sets `props.value` to a result (`result: true`, `action`, `loops`, `before`) and `trigger('submit')` → `editor_done`. The template is static, so setting `props.value` never re-renders the canvas; `watch('value')` picks up new sessions. Loops are closed polylines in image px (pixel centre = c + 0.5), filled even-odd (`rasterize`, PIL). Reshaping (`edited_mask`) uses the new drawing exactly inside the changed zone (`rasterize(before) ^ rasterize(after)` dilated 2 px) and keeps the original pixels elsewhere, so untouched edges stay pixel-exact. Never XOR the rasterizations straight into `lab == r`: the 1-px mismatch between a rasterized outline and the real pixels left specks along the old edge (seen in gauguin_3). After every edit `absorb_specks` merges pieces < 30 px near the change into the region around them; for that, `region_outline` pre-densifies points to < spacing/3 so the JS never adds points to an untouched outline. `add_shape` → new region in a new front group; `reshape_region` → freed pixels go to the nearest other region |
| `history.py` | Undo/redo: snapshot (regions + groups) before each edit in `<project>/history/*.npz` + `index.json`; no-op edits leave no step; max 40; `set_image` wipes it |
| `project.py` | `Project` = one folder. `DEFAULTS` (settings), `GROUP_DEFAULTS` (per kind), `STYLES`, `STYLE_HELP`. Board geometry: `board()` → (width_cm, height_cm, pixels_per_cm) |
| `segment.py` | SAM auto masks → `partition` (big masks first, small on top; uncovered areas split with Felzenszwalb; regions < `min_region_pct` merged) → `auto_groups` (background = regions with ≥25 % of their outline on the image border and ≥ `background_min_pct` area; similar backgrounds merge when ΔE < 18; the rest become objects, clustered by adjacency). Also `mask_from_click` / `add_region_from_mask` for the "add missing region" mode |
| `groups.py` | Group edits. **Merging *fuses* regions in `regions.npy`** (relabels them to the lowest id), so the borders between them disappear. This rewrites the label map, so it can't be undone except by "Reset regions & groups" (re-partitions from cached SAM masks) |
| `layout.py` | `build_layout`: for each group (front first) build a barrier (group boundary + image frame, plus inner region borders for `echo`), take its distance transform, and draw **level lines at (k+½)·spacing** as outline rows. The fill style then adds more echo rows, straight rows (`brick`/`grid`/`rows` at `angle`), or arcs (`fan`, around `fan_center` given as a fraction of the board). Lines are placed longest-first, and runs shorter than `min_run` are dropped. Then `fill_gaps` and optional `relax_sweep` "snugging" |
| `geometry.py` | Shared geometry: resampling, corner detection (60–120° turns kept sharp, everything else smoothed), `line_tiles`, exact rotated-square SAT overlap test, `Placer` (spatial hash, try_add/remove), `relax_sweep` |
| `colors.py` | Bilateral smoothing → median colour of a 4×4 sample grid under each tile → saturation boost in Lab → **k-means palette per group** (`colors` count) → outline contrast (row-0 tiles near a group border pick a lighter or darker palette colour, whichever contrasts with the colour across the border). `export()` writes the CSVs |
| `render.py` | PIL-only drawing: `regions_overlay` (tinted groups, numbered regions, group labels), `draw_tiles`. The overlay's selection-independent part is cached under a key from `pipeline.preview_regions` (regions.npy mtime + groups), and label points use bounding-box crops subsampled ×3: a region click costs ~35 ms server-side (was ~600). The overlay is sent as JPEG and `preview_1_regions.png` is only written when nothing is selected |
| `pipeline.py` | Glue: `build()` (layout → lines preview → colours → exports → `mosaic.png` + `tiles.json`), `run_all()` (segment if needed, then build), `is_built()` |

**Progress convention:** long functions take `progress(frac in [0,1], stage_text)` (default `segment.no_progress`; `segment.scaled(progress, a, b)` maps a sub-step onto part of the bar). SAM has no progress of its own: `segment._report_batches` wraps the generator's `predictor.set_image` and `_process_batch` on the instance.

### Fill styles (`project.STYLES`)
`echo` (rows follow the outline all the way in; objects default to it) · `brick` (straight
rows, half-tile offset; backgrounds default to it) · `grid` · `rows` (straight, packed from
the start of each run) · `fan` (concentric arcs, opus circumactum).

### Key conventions and gotchas
- **Coordinates:** all geometry is in **cm** with the **origin at the bottom-left** of the
  board (y points up). Images are (row, col) with row 0 at the top. Convert with
  `geometry.to_board` and `_lookup`, which also does point→pixel lookup. The board height
  comes from the image aspect ratio.
- **Spacing** = `tile_cm + grout_cm`. Every tile in a project has the same size.
- **Angles:** radians internally and in `tiles.json` (unbounded). `tiles.csv` writes
  `degrees % 90`, because a square looks the same after a 90° turn. The research `Tile`
  class in `mosaic.py` likewise stores degrees mod 90.
- **`tile["group"]` is an index into `project.ordered_groups()`, not the group `id`.** Same
  for `group_map()`. The CSVs write the group *name*.
- **Group order:** `order` 0 is the front. Front groups are placed first, so they get the clean
  outline where two groups meet. `_renumber` and `move` keep `order` contiguous, and empty
  groups get removed.
- **`row` codes on tiles:** 0..n are outline/echo rows (row 0 gets outline contrast).
  `layout.FROZEN = 10_000` marks straight-style tiles, which snugging never moves. `-1` marks
  gap-fill tiles.
- `Placer` tests overlap with a slightly shrunk square (`slack`), so grazing corners are allowed.
- `Project.image()` and `regions()` reload from disk on every call. That's fine at
  working size ≤ 1200 px.
- `set_image()` deletes `regions.npy`, `sam_masks.npz` and `history/`, and clears groups. The app never overwrites a project from a new picture: it picks a free name (`degas_2`).
- `launch(allowed_paths=[PROJECTS])` is needed, or Gradio refuses to serve results from a projects dir outside the cwd.
- Testing the app headlessly: launch it on another port with `app.PROJECTS` pointed at a copy of a project and drive it with `gradio_client` (endpoints are named after the handler functions, e.g. `/open_existing`, `/new_from`, `/undo`, `/build`; State inputs are handled by the client session). Clicks on the picture (`gr.SelectData`) can't be sent this way. For those and for the editor, use Playwright in a separate venv with the system Chrome (`chromium.launch(channel="chrome", headless=True)`); the editor's image→page mapping follows `fit()` in editor.js (z = 0.97·min(stage_w/W, stage_h/H), centred).
- `pipeline.preview_lines` has a leftover no-op `if … : pass` block. It's harmless.

### Project folder layout (`mosaic_projects/<name>/`)
`image.png` (working copy, longest side ≤ `working_size`) · `project.json` (settings + groups,
hand-editable) · `sam_masks.npz` · `regions.npy` (int32 label map, regions numbered largest
first) · `history/` (undo steps) · `preview_1_regions.png` · `preview_2_lines.png` · `mosaic.png` · `tiles.json` ·
`tiles.csv` (`id,x_cm,y_cm,angle_deg,size_cm,color_hex,r,g,b,group`) · `shopping_list.csv`
(`color_hex,r,g,b,tiles,used_in`, sorted by count).

Existing projects: `degas`, `matisse`, `rousseau`, `venus`, `venus2`, `vermeer`, `IMG_0734`.
`degas` is the main test image (30 cm board, 0.2 cm tiles, 2 snug rounds; groups "dancers"
(echo), "wall" (brick −28°), "floor"…).

Group schema in `project.json`:
`{id, name, kind: object|background, members: [region ids], order, style, angle (deg), outline_rows, inner_detail (0–1, echo only), colors, fan_center: [fx, fy]}`

---

## Research scripts (root level), for reference

They import each other by bare module name, so run them from the repo root. Most read
`degas.jpg` and/or precomputed edge maps in `edge_maps/*.npy`. Those edge maps (SAUGE ViT-L,
HED, PiDiNet, TEED, etc.) were generated **outside this repo**, and no code here produces them.

| Approach | Files | Outcome |
|---|---|---|
| Data model and drawing | `mosaic.py` (`Tile`, `Mosaic`), `visualize.py`, `example.py` | Used by all the research scripts, not by the package |
| Differentiable optimisation (PyTorch) | `loss.py` (overlap, correspondence, alignment, neighbour, `total_loss`, `NeighbourList`), `demo_loss.py`, `demo_correspondence.py`, `demo_grid.py` | Gradient descent on positions, angles, and colours. Superseded |
| Direction fields | `gvf.py` (Gradient Vector Flow, Battiato 2008), `edge_field.py` (edge map → skeleton → pruned lines → structure-tensor direction → "echo" field), `demo_edge_field.py`, `demo_edge_grid.py`, `demo_frame.py`, `demo_neighbour.py` | Edge skeleton ideas carried over |
| Deformed grid | `deformed_grid.py` (Instant Field-Aligned Meshes position field, 2D), `demo_deformed_grid.py`, `small_tiles.py`, `new_degas.py` | Superseded |
| Neural net | `nn_mosaic.py` (`MosaicNet`: per-cell presence, offset, angle, colour), `train_nn.py`, `compare_nn.py`, `compare_small.py` | Superseded. Weights in `small_tiles/network/mosaicnet.pt` |
| **Guide lines (andamento)** | `guide_lines.py` (`build_line_mosaic`), `snug.py`, `tile_colors.py`, `demo_lines.py`, `demo_fill.py`, `demo_rows.py`, `demo_snug.py`, `demo_colors.py` | **The winner.** Became `mosaic_studio/geometry.py`, `layout.py`, and `colors.py`. The package swapped edge-map contours for SAM regions |
| SAM prototype | `pipeline_step1_segment.py` → `pipeline/` | Became `mosaic_studio/segment.py` |

`geometry.py` largely **duplicates** functions from `guide_lines.py` and `snug.py`. When
fixing geometry, edit the package copy. The root copies are frozen.

---

## Working on this project

- Smoke test after changes: `python -m mosaic_studio run degas.jpg --name _test --tile 0.6`.
  It reuses nothing because it's a new project, so SAM runs. To skip SAM, rebuild an existing
  project instead: `python -c "from mosaic_studio.project import Project; from mosaic_studio import pipeline; pipeline.build(Project('mosaic_projects/degas'))"`.
  Note that this overwrites that project's outputs.
- Code style: compact numpy/scipy, short names (`lab`, `gmap`, `ppc`, `gi`), `from __future__ import annotations`,
  docstrings explaining the *why* in plain language, section separators `# ----- #`. Heavy
  imports (`torch`, `segment_anything`, `cv2`) are imported lazily inside functions.
- The UI text and docs are in English, though the user is French (folder name "mosaique").
