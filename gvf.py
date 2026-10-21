"""Gradient Vector Flow (GVF) field of an image.

Follows Battiato et al., "Artificial Mosaics by Gradient Vector Flow" (Eurographics 2008),
which uses the GVF of Xu & Prince (1998):

    1. L  = luminance of the image
    2. G  = Roberts gradient of the histogram-equalized L;  f = |G|, scaled to [0, 1]
    3. (u, v) minimizes  E = ∫∫ mu·(u_x² + u_y² + v_x² + v_y²) + |∇f|²·|(u, v) - ∇f|²

Near edges the field ≈ ∇f (perpendicular to the edge). Away from edges the
diffusion term spreads that direction smoothly into flat regions. In the paper each
tile is placed with its sides parallel to the field.

This is a PRE-PROCESSING step: compute it once per image, then pass it to
`loss.alignment_loss`. It is not differentiated through.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def luminance(image: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) RGB in [0, 1] -> (H, W) luminance."""
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def equalize(x: torch.Tensor) -> torch.Tensor:
    """Histogram equalization: each value is replaced by its rank, scaled to [0, 1]."""
    flat = x.reshape(-1)
    ranks = torch.empty_like(flat)
    ranks[flat.argsort()] = torch.arange(flat.numel(), dtype=x.dtype, device=x.device)
    return (ranks / (flat.numel() - 1)).reshape(x.shape)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a (H, W) tensor; sigma in pixels (0 = no blur)."""
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x[None, None], (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[0, 0]


def roberts_magnitude(x: torch.Tensor) -> torch.Tensor:
    """|Roberts cross gradient| of a (H, W) tensor (same size, last row/col padded)."""
    xp = F.pad(x[None, None], (0, 1, 0, 1), mode="replicate")[0, 0]
    g1 = xp[:-1, :-1] - xp[1:, 1:]
    g2 = xp[1:, :-1] - xp[:-1, 1:]
    return torch.sqrt(g1 ** 2 + g2 ** 2)


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    xp = F.pad(x[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    return xp[:-2, 1:-1] + xp[2:, 1:-1] + xp[1:-1, :-2] + xp[1:-1, 2:] - 4 * x


def compute_gvf(image: torch.Tensor, mu: float = 0.2, iterations: int = 1000,
                max_size: int = 300, blur_sigma: float = 1.0) -> torch.Tensor:
    """GVF field of an image, as an (h, w, 2) tensor in MOSAIC directions.

    image      : (H, W, 3) RGB in [0, 1]
    mu         : smoothness (more = direction spreads further into flat areas)
    iterations : diffusion steps (more = spreads further)
    max_size   : the image is first shrunk so its longest side is <= max_size
                 pixels, which makes the diffusion much faster
    blur_sigma : light blur (pixels) before the gradient, to ignore brushstroke noise

    Returned vectors are (x right, y UP), matching mosaic coordinates, and scaled
    so the largest one has length 1. The grid covers the whole board, like the
    image does, so it can be sampled with `loss.sample_image`.
    """
    img = image.permute(2, 0, 1)[None]
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1:
        img = F.interpolate(img, scale_factor=scale, mode="area")
    img = img[0].permute(1, 2, 0)

    lum = gaussian_blur(equalize(luminance(img)), blur_sigma)
    f = roberts_magnitude(lum)
    f = f / (f.max() + 1e-12)                          # edge map in [0, 1]

    fy, fx = torch.gradient(f)                          # rows (down), columns (right)
    mag2 = fx ** 2 + fy ** 2
    u, v = fx.clone(), fy.clone()
    dt = 1.0 / (4 * mu + mag2.max().item() + 1e-12)     # stable explicit step
    for _ in range(iterations):
        u = u + dt * (mu * _laplacian(u) - (u - fx) * mag2)
        v = v + dt * (mu * _laplacian(v) - (v - fy) * mag2)

    field = torch.stack([u, -v], dim=-1)                # image rows point down -> flip y
    return field / (field.norm(dim=-1).max() + 1e-12)
