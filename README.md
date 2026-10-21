# Mosaic Studio

Turns any picture into a mosaic plan: where every tile goes, at what angle, in
what colour, plus a shopping list of tiles per colour.

## Install (once)

```bash
source .venv/bin/activate          # your virtual environment
pip install -r requirements.txt
```

The first segmentation downloads the Segment Anything weights (vit_l ≈ 1.2 GB,
vit_b ≈ 375 MB) into `~/.cache/mosaic_studio/`.

## Use the app

Double-click **`Mosaic Studio.command`**, or run:

```bash
python -m mosaic_studio app
```

A page opens in your browser (http://127.0.0.1:7860). Projects are saved in
`mosaic_projects/<name>/`, so you can close it and reopen later.

The **status line** at the top always says what is going on: during long steps
(finding regions, building) it shows the current stage, a progress bar and the
time so far; when a step is done it says so and what to do next, and opens the
right tab. **↶ Undo / ↷ Redo** next to it take back any change to regions,
groups or styles (they survive closing the app).

1. **Picture**: choose a picture, board width, tile size and grout width, then
   press **Find regions →**. Or press **Do everything automatically**. To pick
   up where you left off, choose a project under *…or continue a project*.
2. **Regions**: the picture is split into numbered regions, collected in groups
   (names on the picture, one colour tint each). Each group is tiled as one shape.
   - ① Click regions to select them (or switch to *Add a missing shape* and click
     on something the regions missed: SAM guesses its outline).
   - **✏️ Draw new shape**: outline a shape yourself on the plain picture. Drag to
     draw; stop whenever you like and the line keeps a blue dot at each open end;
     drag from a dot to carry on, end on a dot to connect or close. ✂️ Erase removes
     a segment. Zoom with pinch, ⌘/Ctrl + scroll or + −; move with scroll or
     Space + drag; ⌘Z undoes a stroke. **✓ Confirm shape** adds it as a new group
     at the front.
   - **✏️ Modify selected shape** (select one region first): the same editor with
     the region's outline already drawn, cut into short segments. Erase the wrong
     segments and redraw that part between the two blue dots. Area the shape gains
     comes from its neighbours; area it loses goes to the nearest neighbour.
   - ② Put the selection in a group: **Add to this group**, or **Create new
     group**. The regions are fused into one shape with the group: the lines
     between them disappear and the rows flow through.
   - ③ Groups, front to back: where two groups touch, the front one keeps the
     clean outline. **▲ / ▼** reorder, and you can rename a group, switch it between
     object and background, or select all its regions.
   - *Start over* recomputes the automatic regions and groups (quick; undoable).
3. **Styles & build**: for each group pick a fill style: *echo* (rows follow the
   outline all the way in), *brick*, *grid*, *rows* (all with an angle), or *fan*
   (arcs around a point), plus outline rows and colours. Changes are saved as you
   make them. Then **Build mosaic →**.
4. **Result**: the mosaic, the downloads and the shopping list.

## Automatic, no clicking

```bash
python -m mosaic_studio run photo.jpg --board-width 40 --tile 0.5
```

## Outputs (in the project folder)

- `mosaic.png`: the rendered mosaic
- `tiles.csv`: one line per tile: position of its centre in cm (origin
  bottom-left), angle in degrees, size, colour, group
- `shopping_list.csv`: number of tiles of each colour
- `project.json`: every setting and group, editable by hand
- `preview_1_regions.png`, `preview_2_lines.png`: previews of the steps

## Speed

Segmentation is the slow part: a few minutes on a laptop CPU the first time
(the smaller `vit_b` model is faster). After that it is cached; rebuilding with
new styles takes about a minute for ~6 000 tiles.
