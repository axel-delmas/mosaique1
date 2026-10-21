"""Draw a Mosaic with matplotlib.

Usage from code:
    from visualize import show_mosaic
    show_mosaic(my_mosaic)                      # opens a window
    show_mosaic(my_mosaic, save_to="out.png")   # saves an image instead
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from mosaic import Mosaic


def draw_mosaic(mosaic: Mosaic, ax=None, background=(0.85, 0.83, 0.78),
                grout=(0.3, 0.3, 0.3), grout_width=0.5, show_border=True):
    """Draw the mosaic onto a matplotlib Axes and return that Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8 * mosaic.height / mosaic.width))

    # Board background (the "mortar")
    ax.add_patch(Rectangle((0, 0), mosaic.width, mosaic.height,
                           facecolor=background, edgecolor="black" if show_border else "none",
                           linewidth=1.5, zorder=0))

    # all tiles in one collection: fast even for tens of thousands of tiles
    if mosaic.tiles:
        polys = [tile.corners(mosaic.tile_width) for tile in mosaic.tiles]
        faces = [tuple(c / 255 for c in tile.color) for tile in mosaic.tiles]
        ax.add_collection(PolyCollection(polys, facecolors=faces, edgecolors=[grout],
                                         linewidths=grout_width, zorder=1))

    ax.set_xlim(0, mosaic.width)
    ax.set_ylim(0, mosaic.height)
    ax.set_aspect("equal")
    u = mosaic.unit
    ax.set_xlabel(f"x ({u})")
    ax.set_ylabel(f"y ({u})")
    ax.set_title(f"Mosaic {mosaic.width:g} x {mosaic.height:g} {u} — "
                 f"{len(mosaic)} tiles of {mosaic.tile_width:g} {u}")
    return ax


def show_mosaic(mosaic: Mosaic, save_to: str | None = None, **kwargs) -> None:
    """Draw the mosaic and either show it in a window or save it to a file."""
    ax = draw_mosaic(mosaic, **kwargs)
    fig = ax.figure
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
        plt.close(fig)
    else:
        plt.show()
