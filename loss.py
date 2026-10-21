"""Differentiable losses on tile layouts (PyTorch).

The losses work on plain tensors, not on Mosaic objects, so they can sit inside
a training loop. Use `mosaic_to_tensors` / `tensors_to_mosaic` to go back and forth.

Conventions
-----------
positions    : (N, 2) tensor of tile centers, in the mosaic's unit
orientations : (N,) tensor of angles in RADIANS, unconstrained
               (wrap with `% (pi/2)` only for display; the Tile class does this itself)
colors       : (N, 3) tensor of RGB in [0, 1]
image        : (H, W, 3) tensor of RGB in [0, 1], stretched over the whole board
field        : (h, w, 2) GVF field from `gvf.compute_gvf`, also stretched over the board
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mosaic import Mosaic, Tile


# --------------------------------------------------------------------------- #
# Mosaic <-> tensors
# --------------------------------------------------------------------------- #
def mosaic_to_tensors(mosaic: Mosaic, requires_grad: bool = True,
                      dtype=torch.float32, device=None):
    """Return (positions (N,2), orientations (N,) in radians, colors (N,3) in [0,1])
    as leaf tensors. Colors never require grad here; call
    `colors.requires_grad_()` if you want to optimize them too."""
    pos = torch.tensor([t.position for t in mosaic.tiles], dtype=dtype, device=device)
    ori = torch.tensor([math.radians(t.orientation) for t in mosaic.tiles],
                       dtype=dtype, device=device)
    col = torch.tensor([t.color for t in mosaic.tiles], dtype=dtype, device=device) / 255
    return pos.requires_grad_(requires_grad), ori.requires_grad_(requires_grad), col


def tensors_to_mosaic(mosaic: Mosaic, positions: torch.Tensor,
                      orientations: torch.Tensor | None = None,
                      colors: torch.Tensor | None = None) -> Mosaic:
    """Return a NEW Mosaic with the same board and the given positions/angles/colors.
    Anything left as None is copied from `mosaic`."""
    pos = positions.detach().cpu().tolist()
    ori = (orientations.detach().cpu().tolist() if orientations is not None
           else [math.radians(t.orientation) for t in mosaic.tiles])
    col = ([tuple(round(c * 255) for c in rgb)
            for rgb in colors.detach().clamp(0, 1).cpu().tolist()]
           if colors is not None else [t.color for t in mosaic.tiles])
    tiles = [Tile(color=c, position=tuple(p), orientation=math.degrees(o))
             for p, o, c in zip(pos, ori, col)]
    return Mosaic(mosaic.width, mosaic.height, mosaic.tile_width, tiles, mosaic.unit)


def load_image(path: str, dtype=torch.float32, device=None) -> torch.Tensor:
    """Read an image file into an (H, W, 3) tensor with values in [0, 1]."""
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 255
    return torch.tensor(arr, dtype=dtype, device=device)


def orientation_features(orientations: torch.Tensor) -> torch.Tensor:
    """(N,) radians -> (N, 2) [cos 4θ, sin 4θ]: same value for θ and θ + 90°.
    This is what a neural net should see instead of the raw angle."""
    return torch.stack([torch.cos(4 * orientations), torch.sin(4 * orientations)], dim=-1)


# --------------------------------------------------------------------------- #
# Neighbour list (fixed radius + skin, rebuilt only when tiles moved enough)
# --------------------------------------------------------------------------- #
class NeighbourList:
    """Candidate pairs of tiles for the pair losses, without checking all N² pairs.

    Pairs closer than radius + 2·skin are stored. The list stays valid (no pair
    that could get closer than `radius` is missing) as long as no tile has moved
    more than `skin` since the list was built: two tiles can then have closed the
    gap by at most 2·skin. When a tile moves further, the whole list is rebuilt
    (a k-d tree query, fast even for tens of thousands of tiles).

    radius: the largest distance any pair loss looks at, e.g.
            max(contact_factor, radius_factor) * tile_width
    skin:   the ε margin; ~0.25 tile is a good default
    """

    def __init__(self, radius: float, skin: float):
        self.radius, self.skin = radius, skin
        self.ref = None
        self.idx = None
        self.rebuilds = 0

    def pairs(self, positions: torch.Tensor):
        p = positions.detach()
        if self.ref is None or (p - self.ref).norm(dim=-1).max() > self.skin:
            from scipy.spatial import cKDTree
            ij = cKDTree(p.cpu().numpy()).query_pairs(self.radius + 2 * self.skin,
                                                      output_type="ndarray")
            ij = torch.as_tensor(ij, dtype=torch.long, device=positions.device)
            self.idx = (ij[:, 0], ij[:, 1]) if len(ij) else (ij.new_zeros(0), ij.new_zeros(0))
            self.ref = p.clone()
            self.rebuilds += 1
        return self.idx


# --------------------------------------------------------------------------- #
# Overlap loss (naive O(N^2))
# --------------------------------------------------------------------------- #
def overlap_penalty(d: torch.Tensor, tile_width: float,
                    contact_factor: float = math.sqrt(2),
                    sharpness: float = 4.0) -> torch.Tensor:
    """Penalty for two tiles whose centers are a distance d apart.

        contact = contact_factor * tile_width      (default √2·w)
        u       = max(0, contact - d) / tile_width (how far inside, in tile widths)
        f(u)    = exp(k·u) - 1 - k·u               (k = sharpness)

    f is 0 for d >= contact, and both f and its slope are 0 at the contact
    distance, so the loss is smooth when a pair starts/stops touching.
    It then grows exponentially as the tiles get closer. Dividing by
    tile_width makes `sharpness` independent of the unit (cm, mm, ...).
    """
    contact = contact_factor * tile_width
    u = torch.relu(contact - d) / tile_width
    k = sharpness
    return torch.expm1(k * u) - k * u


def overlap_loss(positions: torch.Tensor, tile_width: float,
                 contact_factor: float = math.sqrt(2),
                 sharpness: float = 4.0,
                 reduction: str = "sum", pairs=None) -> torch.Tensor:
    """`overlap_penalty` over ALL N(N-1)/2 pairs of tiles, or only over `pairs`
    (an (i, j) pair of index tensors, e.g. from NeighbourList) when given.

    reduction: "sum" (total), "per_tile" (sum / N), "mean" (mean over pairs)
               or "none" (one value per pair).
    Brute force without `pairs`: fine up to a few thousand tiles. Depends only on
    the centers, so it does not constrain orientation (its gradient w.r.t. angles is 0).
    """
    n = positions.shape[0]
    if n < 2:
        return positions.sum() * 0.0
    i, j = pairs if pairs is not None else torch.triu_indices(n, n, offset=1,
                                                             device=positions.device)
    diff = positions[i] - positions[j]
    # small constant inside the sqrt: the gradient of |x| is undefined at x = 0,
    # which would give NaNs if two tiles sit exactly on top of each other
    d = torch.sqrt((diff * diff).sum(-1) + 1e-12 * tile_width ** 2)
    pen = overlap_penalty(d, tile_width, contact_factor, sharpness)
    if reduction == "sum":
        return pen.sum()
    if reduction == "per_tile":
        return pen.sum() / n
    if reduction == "mean":
        return pen.mean() if pairs is None else pen.sum() / (n * (n - 1) / 2)
    if reduction == "none":
        return pen
    raise ValueError(f"unknown reduction {reduction!r}")


# --------------------------------------------------------------------------- #
# Correspondence loss
# --------------------------------------------------------------------------- #
def tile_sample_points(positions: torch.Tensor, orientations: torch.Tensor,
                       tile_width: float, samples_per_side: int = 4) -> torch.Tensor:
    """(N, S, 2) points spread evenly INSIDE each tile (S = samples_per_side²),
    in mosaic coordinates. They follow the tile's position and rotation, so
    anything computed from them is differentiable w.r.t. both."""
    k = samples_per_side
    # cell centers of a k x k grid over [-w/2, w/2]^2
    t = (torch.arange(k, dtype=positions.dtype, device=positions.device) + 0.5) / k - 0.5
    gx, gy = torch.meshgrid(t, t, indexing="xy")
    local = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1) * tile_width  # (S, 2)
    c, s = torch.cos(orientations), torch.sin(orientations)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (N, 2, 2)
    return positions[:, None, :] + torch.einsum("nij,sj->nsi", rot, local)


def sample_image(image: torch.Tensor, points: torch.Tensor,
                 board_width: float, board_height: float) -> torch.Tensor:
    """Bilinear lookup of image values at mosaic-coordinate points.

    image  : (H, W, C), stretched so it exactly covers the board
             (image top row = top of the board, y = board_height).
             Works for any number of channels C (RGB image, GVF field, ...).
    points : (..., 2) in mosaic coordinates
    returns: (..., C)
    Points outside the board get the color of the nearest image edge.
    """
    shape = points.shape[:-1]
    x, y = points[..., 0], points[..., 1]
    # grid_sample wants coordinates in [-1, 1], with y pointing DOWN
    grid = torch.stack([2 * x / board_width - 1, 1 - 2 * y / board_height], -1)
    out = F.grid_sample(image.permute(2, 0, 1)[None],          # (1, C, H, W)
                        grid.reshape(1, 1, -1, 2),
                        mode="bilinear", padding_mode="border", align_corners=False)
    return out[0, :, 0, :].T.reshape(*shape, image.shape[-1])


def correspondence_loss(positions: torch.Tensor, orientations: torch.Tensor,
                        colors: torch.Tensor, image: torch.Tensor,
                        tile_width: float, board_width: float, board_height: float,
                        samples_per_side: int = 4) -> torch.Tensor:
    """How far the mosaic's colors are from the image, where tiles are.

    For each tile, the image is sampled at S points inside the tile and each
    sample is compared to the tile's (flat) color with a squared L2 distance
    over RGB. Only points covered by tiles are counted, so grout and empty
    board areas are ignored.

    Returns the mean over tiles and samples (0 = perfect match, max 3).
    Differentiable w.r.t. positions, orientations (through the sample points)
    and colors.
    """
    pts = tile_sample_points(positions, orientations, tile_width, samples_per_side)
    target = sample_image(image, pts, board_width, board_height)       # (N, S, 3)
    return ((target - colors[:, None, :]) ** 2).sum(-1).mean()


# --------------------------------------------------------------------------- #
# Alignment loss (Gradient Vector Flow)
# --------------------------------------------------------------------------- #
def alignment_loss(positions: torch.Tensor, orientations: torch.Tensor,
                   field: torch.Tensor, tile_width: float,
                   board_width: float, board_height: float,
                   samples_per_side: int = 4, softness: float | None = None,
                   weighting: str = "soft") -> torch.Tensor:
    """How badly each tile's sides line up with the image's GVF field.

    Following Battiato et al. 2008, a tile is well placed when its sides are
    parallel to the GVF direction φ under it. A square looks the same every 90°,
    so the mismatch is measured with 4·(θ - φ):

        mismatch = (1 - cos 4(θ - φ)) / 2      0 = aligned, 1 = off by 45°

    This also makes φ and φ + 180° equivalent, which matters because the GVF
    flips direction across an edge.

    field: any (h, w, 2) direction field stretched over the board: the GVF
    (gvf.compute_gvf) or the edge-line field (edge_field.build_edge_field).

    weighting="soft" (for GVF): each sample point is weighted by
    m / (m + softness), with m = |field| there:
    close to 1 near edges, close to 0 in flat areas where the direction means
    little. softness defaults to the median |GVF| of the image. The weight is
    NOT differentiated, so tiles are not rewarded for fleeing strong edges.

    weighting="magnitude" (for edge_field): the weight is |field| itself, which
    edge_field already sets to exp(-distance to nearest line / echo).

    Returns the weighted mean over tiles and sample points, in [0, 1].
    Differentiable w.r.t. orientations and positions.
    """
    if weighting not in ("soft", "magnitude"):
        raise ValueError(f"unknown weighting {weighting!r}")
    if weighting == "soft" and softness is None:
        softness = field.norm(dim=-1).median().item()
    pts = tile_sample_points(positions, orientations, tile_width, samples_per_side)
    g = sample_image(field, pts, board_width, board_height)          # (N, S, 2)
    gx, gy = g[..., 0], g[..., 1]
    r2 = gx * gx + gy * gy
    # (cos 4φ, sin 4φ) from the vector without atan2 (which is unstable near 0):
    # (gx + i·gy)^4 = |g|^4 · e^{4iφ}
    a, b = gx * gx - gy * gy, 2 * gx * gy                             # |g|² e^{2iφ}
    cos4phi = (a * a - b * b) / (r2 * r2 + 1e-12)
    sin4phi = (2 * a * b) / (r2 * r2 + 1e-12)
    th = 4 * orientations[:, None]
    cos_diff = torch.cos(th) * cos4phi + torch.sin(th) * sin4phi       # cos 4(θ - φ)
    m = torch.sqrt(r2).detach()
    weight = m / (m + softness) if weighting == "soft" else m
    return (weight * (1 - cos_diff) / 2).sum() / (weight.sum() + 1e-12)


# --------------------------------------------------------------------------- #
# Neighbour loss (tiles aligned with each other, forming rows)
# --------------------------------------------------------------------------- #
def neighbour_loss(positions: torch.Tensor, orientations: torch.Tensor,
                   tile_width: float, radius_factor: float = math.sqrt(2),
                   row_weight: float = 0.5, pairs=None) -> torch.Tensor:
    """How badly each tile is aligned with its close neighbours.

    Neighbours are all tiles whose centers are closer than radius_factor * w
    (default √2·w, the distance under which two squares can touch). For each pair:

      same angle   (1 - cos 4(θi - θj)) / 2
                   0 when both are rotated the same way (mod 90°)
      side by side (1 - cos 4(φ - θi)) / 2, averaged over i and j,
                   where φ is the direction from i to j: 0 when j sits along
                   one of i's axes (in the same row or column), 1 when diagonal

    mismatch = (1 - row_weight) * same_angle + row_weight * side_by_side

    Pairs are weighted by (1 - d/R)² (closer neighbours count more, smoothly 0
    at R). The weight is NOT differentiated, so tiles are not rewarded for simply
    moving away from misaligned neighbours. Returns the weighted mean, in [0, 1].
    Naive O(N²) over pairs like overlap_loss, unless `pairs` (from NeighbourList)
    is given.
    """
    n = positions.shape[0]
    if n < 2:
        return positions.sum() * 0.0
    R = radius_factor * tile_width
    i, j = pairs if pairs is not None else torch.triu_indices(n, n, offset=1,
                                                             device=positions.device)
    diff = positions[j] - positions[i]
    d = torch.sqrt((diff * diff).sum(-1) + 1e-12 * tile_width ** 2)
    close = d < R
    i, j, diff, d = i[close], j[close], diff[close], d[close]
    if i.numel() == 0:
        return positions.sum() * 0.0
    weight = ((1 - d / R) ** 2).detach()

    ti, tj = orientations[i], orientations[j]
    same_angle = (1 - torch.cos(4 * (ti - tj))) / 2

    ux, uy = diff[:, 0] / d, diff[:, 1] / d
    a, b = ux * ux - uy * uy, 2 * ux * uy          # (cos 2φ, sin 2φ)
    c4, s4 = a * a - b * b, 2 * a * b               # (cos 4φ, sin 4φ)
    side_i = (1 - (c4 * torch.cos(4 * ti) + s4 * torch.sin(4 * ti))) / 2
    side_j = (1 - (c4 * torch.cos(4 * tj) + s4 * torch.sin(4 * tj))) / 2
    side = (side_i + side_j) / 2

    mismatch = (1 - row_weight) * same_angle + row_weight * side
    return (weight * mismatch).sum() / (weight.sum() + 1e-12)


# --------------------------------------------------------------------------- #
# Total loss
# --------------------------------------------------------------------------- #
def total_loss(positions: torch.Tensor, orientations: torch.Tensor,
               colors: torch.Tensor, image: torch.Tensor, mosaic: Mosaic,
               alpha: float = 0.5, beta: float = 0.0,
               field: torch.Tensor | None = None, delta: float = 0.0,
               return_parts: bool = False, pairs=None, **kwargs):
    """(1 - alpha) * overlap + alpha * correspondence + beta * alignment
       + delta * neighbour.

    alpha in [0, 1] trades overlap against matching the image colors
        (0 -> only avoid overlaps, 1 -> only match colors).
    beta >= 0 adds the alignment term on top (0 = orientation ignored).
        It needs a direction `field` computed once beforehand:
        gvf.compute_gvf(image) or edge_field.build_edge_field(edge_map, ...).
    delta >= 0 weighs the neighbour term (tiles aligned with each other / in rows).
    The overlap term uses reduction="per_tile" so no term grows with the number
    of tiles, which keeps alpha and beta meaningful across mosaics.
    kwargs go to the sub-losses (contact_factor, sharpness, samples_per_side,
    softness, weighting, radius_factor, row_weight). With an edge_field field,
    pass weighting="magnitude".

    pairs: optional (i, j) index tensors of candidate pairs (NeighbourList), used
    by the overlap and neighbour terms instead of all N² pairs.

    With return_parts=True, returns (total, overlap, correspondence, alignment,
    neighbour).
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if beta < 0 or delta < 0:
        raise ValueError("beta and delta must be >= 0")
    if beta > 0 and field is None:
        raise ValueError("beta > 0 needs a direction field (gvf or edge_field)")
    ov_kw = {k: kwargs[k] for k in ("contact_factor", "sharpness") if k in kwargs}
    co_kw = {k: kwargs[k] for k in ("samples_per_side",) if k in kwargs}
    al_kw = {k: kwargs[k] for k in ("samples_per_side", "softness", "weighting") if k in kwargs}
    ov = overlap_loss(positions, mosaic.tile_width, reduction="per_tile", pairs=pairs, **ov_kw)
    co = correspondence_loss(positions, orientations, colors, image, mosaic.tile_width,
                             mosaic.width, mosaic.height, **co_kw)
    if field is not None:
        al = alignment_loss(positions, orientations, field, mosaic.tile_width,
                            mosaic.width, mosaic.height, **al_kw)
    else:
        al = positions.sum() * 0.0
    nb_kw = {k: kwargs[k] for k in ("radius_factor", "row_weight") if k in kwargs}
    nb = (neighbour_loss(positions, orientations, mosaic.tile_width, pairs=pairs, **nb_kw)
          if (delta > 0 or return_parts) else positions.sum() * 0.0)
    total = (1 - alpha) * ov + alpha * co + beta * al + delta * nb
    return (total, ov, co, al, nb) if return_parts else total
