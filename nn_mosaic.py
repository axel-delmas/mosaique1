"""A CNN that outputs a mosaic: one potential tile per cell of a fine grid.

The board is cut into cells of size `cell` (half a tile spacing by default, so
there are ~4x more cells than tiles fit). For every cell the network predicts

    presence  p in [0, 1]    is there a tile in this cell?
    offset    (dx, dy)       where in / around the cell (up to ±1 cell)
    angle     θ              as (cos 4θ, sin 4θ), so 90° turns are free
    color     RGB in [0, 1]

The number of tiles is the number of cells with p > 0.5, so it depends on the
input. Training uses the same losses as before, weighted by presence, plus

    coverage      rewards tiles: 1 - (expected tiles x spacing²) / board area
    decisiveness  mean p(1-p), pushes presences to 0 or 1

Training uses the EXPECTED loss: presences enter as probabilities (overlap of a
pair weighted by p_i p_j, coverage by p_i), and overlap and coverage are both
divided by the number of tiles the board holds at the nominal spacing, so the
trade-off per tile is explicit: a tile is worth adding when its coverage reward
beats its overlap cost. The decisiveness term, ramped up during training, then
pushes every p to 0 or 1. (Randomly sampled presences — Gumbel-sigmoid — were
tried first: the noise kept switching on cells right next to tiles, the overlap
cost exploded and the network switched every tile off.)

Pairs of tiles (overlap, neighbour losses) are only compared between cells at
most `window` cells apart, which is exact because offsets are bounded: a fixed-
radius neighbour list, O(cells) instead of O(tiles²).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import overlap_penalty, sample_image, tile_sample_points
from mosaic import Mosaic, Tile


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class MosaicNet(nn.Module):
    """Input (1, C, 4*gh, 4*gw)  ->  raw outputs (1, 8, gh, gw)."""

    def __init__(self, in_ch: int, width: int = 48):
        super().__init__()
        w = width
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, stride=2, padding=1), nn.GELU(),
        )
        # dilated convs: each cell sees several tiles around it, so neighbouring
        # cells can coordinate (who hosts a tile, which way the row goes)
        self.ctx = nn.ModuleList([nn.Conv2d(w, w, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 1)])
        self.head = nn.Conv2d(w, 8, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = -1.0          # start with p ≈ 0.27 everywhere
            self.head.bias[3] = 1.0           # angle 0 as a start
        self.enc_noise = None

    def forward(self, x):
        h = self.enc(x)
        for conv in self.ctx:
            h = h + F.gelu(conv(h))
        return self.head(h)


def make_input(image: torch.Tensor, field: np.ndarray, gh: int, gw: int) -> torch.Tensor:
    """Stack image, edge-field features and coordinates at 4 px per cell."""
    H, W = 4 * gh, 4 * gw
    img = F.interpolate(image.permute(2, 0, 1)[None], size=(H, W), mode="area")
    f = torch.tensor(field, dtype=torch.float32).permute(2, 0, 1)[None]
    f = F.interpolate(f, size=(H, W), mode="area")
    ang = torch.atan2(f[:, 1], f[:, 0])
    wgt = f.norm(dim=1)
    feat = torch.stack([wgt * torch.cos(4 * ang), wgt * torch.sin(4 * ang), wgt], 1)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    coords = torch.stack([xs, -ys])[None]
    return torch.cat([img * 2 - 1, feat, coords], 1)


# --------------------------------------------------------------------------- #
# Decoding raw outputs into tiles
# --------------------------------------------------------------------------- #
def decode(raw: torch.Tensor, cell: float, board_h: float, max_offset: float = 1.0):
    """raw (1, 8, gh, gw) -> dict of (gh, gw, ...) tensors in mosaic coordinates.
    Grid row 0 is the TOP of the board (like the image)."""
    _, _, gh, gw = raw.shape
    r = raw[0]
    logit = r[0]
    ys = (torch.arange(gh, dtype=raw.dtype) + 0.5) * cell
    xs = (torch.arange(gw, dtype=raw.dtype) + 0.5) * cell
    cy, cx = torch.meshgrid(board_h - ys, xs, indexing="ij")
    off = torch.tanh(r[1:3]) * max_offset * cell
    pos = torch.stack([cx + off[0], cy + off[1]], -1)
    theta = torch.atan2(r[4], r[3]) / 4
    color = torch.sigmoid(r[5:8]).permute(1, 2, 0)
    return dict(logit=logit, pos=pos, theta=theta, color=color)


def gumbel_sigmoid(logit, tau, hard=False):
    u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
    y = torch.sigmoid((logit + torch.log(u) - torch.log1p(-u)) / tau)
    if hard:
        return (y > 0.5).float() - y.detach() + y     # straight-through
    return y


# --------------------------------------------------------------------------- #
# Losses on the cell grid
# --------------------------------------------------------------------------- #
def _shift(t, dy, dx):
    """out[y, x] = t[y + dy, x + dx] (zero outside), plus a validity mask."""
    gh, gw = t.shape[:2]
    out = torch.zeros_like(t)
    valid = torch.zeros(gh, gw, dtype=torch.bool)
    ys, yd = (slice(dy, gh), slice(0, gh - dy)) if dy >= 0 else (slice(0, gh + dy), slice(-dy, gh))
    xs, xd = (slice(dx, gw), slice(0, gw - dx)) if dx >= 0 else (slice(0, gw + dx), slice(-dx, gw))
    out[yd, xd] = t[ys, xs]
    valid[yd, xd] = True
    return out, valid


def pair_losses(pos, theta, z, tile_width, window, contact_factor, sharpness,
                radius_factor=math.sqrt(2), row_weight=0.5, norm=None):
    """Overlap and neighbour terms over all cell pairs within `window` cells.
    Each unordered pair is visited once (half of the shifts).
    Overlap is summed over pairs (weighted z_i z_j) and divided by `norm`
    (default: the current number of tiles, sum z)."""
    R = radius_factor * tile_width
    ov_sum = pos.new_zeros(())
    nb_sum = pos.new_zeros(())
    nb_w = pos.new_zeros(())
    for dy in range(0, window + 1):
        for dx in range(-window, window + 1):
            if dy == 0 and dx <= 0:
                continue
            pj, v = _shift(pos, dy, dx)
            tj, _ = _shift(theta, dy, dx)
            zj, _ = _shift(z, dy, dx)
            diff = pj - pos
            d = torch.sqrt((diff * diff).sum(-1) + 1e-12)
            zz = z * zj * v
            ov_sum = ov_sum + (zz * overlap_penalty(d, tile_width, contact_factor, sharpness)).sum()
            close = (d < R) & v
            if close.any():
                w = ((1 - d / R).clamp(min=0) ** 2).detach() * zz
                same = (1 - torch.cos(4 * (theta - tj))) / 2
                ux, uy = diff[..., 0] / d, diff[..., 1] / d
                a, b = ux * ux - uy * uy, 2 * ux * uy
                c4, s4 = a * a - b * b, 2 * a * b
                side = ((1 - (c4 * torch.cos(4 * theta) + s4 * torch.sin(4 * theta))) / 2 +
                        (1 - (c4 * torch.cos(4 * tj) + s4 * torch.sin(4 * tj))) / 2) / 2
                mis = (1 - row_weight) * same + row_weight * side
                nb_sum = nb_sum + (w * mis * close).sum()
                nb_w = nb_w + (w * close).sum()
    n = z.sum() + 1e-6 if norm is None else norm
    return ov_sum / n, nb_sum / (nb_w + 1e-6)


def tile_losses(pos, theta, color, z, image, field, tile_width, board_w, board_h,
                samples_per_side=4):
    """Correspondence and alignment, averaged over tiles weighted by presence."""
    P = pos.reshape(-1, 2)
    T = theta.reshape(-1)
    Z = z.reshape(-1)
    pts = tile_sample_points(P, T, tile_width, samples_per_side)
    target = sample_image(image, pts, board_w, board_h)
    corr_t = ((target - color.reshape(-1, 1, 3)) ** 2).sum(-1).mean(1)
    corr = (Z * corr_t).sum() / (Z.sum() + 1e-6)

    g = sample_image(field, pts, board_w, board_h)
    gx, gy = g[..., 0], g[..., 1]
    r2 = gx * gx + gy * gy
    a, b = gx * gx - gy * gy, 2 * gx * gy
    cos4p, sin4p = (a * a - b * b) / (r2 * r2 + 1e-12), 2 * a * b / (r2 * r2 + 1e-12)
    cosd = torch.cos(4 * T)[:, None] * cos4p + torch.sin(4 * T)[:, None] * sin4p
    m = torch.sqrt(r2).detach() * Z[:, None]
    align = (m * (1 - cosd) / 2).sum() / (m.sum() + 1e-6)
    return corr, align


def boundary_loss(pos, z, tile_width, board_w, board_h):
    """Penalty for tile centers closer than half a tile to the board edge."""
    h = tile_width / 2
    x, y = pos[..., 0], pos[..., 1]
    out = (F.relu(h - x) + F.relu(x - (board_w - h)) +
           F.relu(h - y) + F.relu(y - (board_h - h))) / tile_width
    return (z * out ** 2).sum() / (z.sum() + 1e-6)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def to_mosaic(out: dict, board_w, board_h, tile_width, threshold=0.5, unit="cm") -> Mosaic:
    keep = torch.sigmoid(out["logit"]) > threshold
    pos = out["pos"][keep].detach().numpy()
    th = out["theta"][keep].detach().numpy()
    col = out["color"][keep].detach().clamp(0, 1).numpy()
    tiles = [Tile(color=tuple(int(round(c * 255)) for c in cc), position=tuple(p),
                  orientation=math.degrees(t)) for p, t, cc in zip(pos, th, col)]
    return Mosaic(board_w, board_h, tile_width, tiles, unit)
