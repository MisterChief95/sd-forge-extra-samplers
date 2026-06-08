# ssprk3.py — SSPRK3 (Strong Stability Preserving Runge-Kutta 3rd Order)
# A stable 3rd-order explicit sampler for Forge/KDiffusionSampler.
#
# SSPRK3 (Shu-Osher form):
#   Stage 1: u1 = x_n + h*k1              where k1 = f(x_n)
#   Stage 2: u2 = (3/4)*x_n + (1/4)*u1 + (1/4)*h*k2   where k2 = f(u1)
#   Stage 3: x_n+1 = (1/3)*x_n + (2/3)*u2 + (2/3)*h*k3   where k3 = f(u2)
#
# This method preserves strong stability properties (SSP) and is particularly
# good for problems requiring both accuracy and stability - ideal for diffusion.

from __future__ import annotations

import torch
from tqdm import trange
from modules import sd_samplers, sd_samplers_common
from modules.sd_samplers_kdiffusion import KDiffusionSampler

from lib_es.utils import sampler_metadata


def _normalize_sigmas(sigmas, device, dtype) -> torch.Tensor:
    if sigmas is None:
        return None
    if not torch.is_tensor(sigmas):
        sigmas = torch.tensor(sigmas, device=device, dtype=dtype)
    else:
        sigmas = sigmas.to(device=device, dtype=dtype)
    if sigmas.ndim != 1:
        sigmas = sigmas.flatten()
    return sigmas.contiguous()


def _to_d(x: torch.Tensor, sigma, denoised: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    else:
        sigma = sigma.to(device=x.device, dtype=x.dtype)

    sigma = sigma.clamp_min(eps)

    # If sigma is per-batch [B], reshape to [B,1,1,1] for NCHW broadcast
    if sigma.ndim == 1:
        sigma = sigma.view(-1, 1, 1, 1)

    return (x - denoised) / sigma


@sampler_metadata("SSPRK3")
@torch.no_grad()
def sample_ssprk3(
    model,
    x: torch.Tensor,
    *,
    sigmas=None,
    extra_args=None,
    callback=None,
    disable=False,
    **kwargs,
):
    """
    SSPRK3 (Strong Stability Preserving Runge-Kutta 3rd Order) sampler.

    This is a 3rd-order explicit method with excellent stability properties.
    It uses 3 model evaluations per step and works well with standard schedules
    (Karras, AYS, SGM Uniform, etc.).

    The SSP property ensures the method maintains stability bounds that would
    hold for forward Euler, making it robust for stiff problems like diffusion.
    """
    # Forge variants sometimes pass schedule under different names
    if sigmas is None:
        sigmas = kwargs.get("sigmas") or kwargs.get("sigma_sched")
    if sigmas is None:
        raise ValueError("[SSPRK3 Sampler] missing sigmas schedule")

    device, dtype = x.device, x.dtype
    sigmas = _normalize_sigmas(sigmas, device, dtype)

    steps = int(sigmas.numel() - 1)
    if steps <= 0:
        return x

    ea = extra_args or {}
    s_in = torch.ones((x.shape[0],), device=device, dtype=dtype)

    def _cb(i: int, sigma: torch.Tensor, sigma_next: torch.Tensor, denoised: torch.Tensor):
        if callback is None:
            return
        try:
            callback({"i": i, "sigma": sigma, "sigma_next": sigma_next, "x": x, "denoised": denoised})
        except TypeError:
            callback(i, x, sigma, sigma_next)

    eps_sigma = torch.tensor(1e-8, device=device, dtype=dtype)

    for i in trange(steps, disable=disable):
        s0 = sigmas[i]
        s1 = sigmas[i + 1]
        h = s1 - s0

        # Stage 1: Evaluate at current point
        den1 = model(x, s0 * s_in, **ea)

        # Handle final step to sigma=0 with simple Euler
        if float(s1) == 0.0:
            d1 = _to_d(x, s0, den1)
            x = x + h * d1
            _cb(i, s0, s1, den1)
            continue

        d1 = _to_d(x, s0, den1)

        # Stage 1: Full Euler step
        u1 = x + h * d1

        # Evaluate at stage 1
        # Sigma for u1: since u1 = x + h*d1, we're at s0 + h = s1
        den2 = model(u1, s1 * s_in, **ea)
        d2 = _to_d(u1, s1, den2)

        # Stage 2: Convex combination
        # u2 = (3/4)*x_n + (1/4)*u1 + (1/4)*h*d2
        u2 = (3.0 / 4.0) * x + (1.0 / 4.0) * u1 + (1.0 / 4.0) * h * d2

        # Sigma for u2: weighted average
        # u2 is roughly at (3/4)*s0 + (1/4)*s1 + (1/4)*(s1-s0) = s0 + (1/2)*(s1-s0)
        s2 = s0 + 0.5 * h
        s2 = torch.maximum(s2, eps_sigma) if torch.is_tensor(s2) else max(float(s2), 1e-8)

        # Evaluate at stage 2
        den3 = model(u2, s2 * s_in, **ea)
        d3 = _to_d(u2, s2, den3)

        # Stage 3: Final convex combination
        # x_n+1 = (1/3)*x_n + (2/3)*u2 + (2/3)*h*d3
        x = (1.0 / 3.0) * x + (2.0 / 3.0) * u2 + (2.0 / 3.0) * h * d3

        # Safety: handle NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x)

        _cb(i, s0, s1, den3)

    return x
