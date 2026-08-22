import math
import torch

from modules_forge.packages.k_diffusion.sampling import to_d

from tqdm.auto import trange

from ..utils import churn_gamma, is_rf_model, rf_churn_step, sampler_metadata


@sampler_metadata("Euler Max")
@torch.no_grad()
def sample_euler_max(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = churn_gamma(s_churn, len(sigmas) - 1, sigmas[i], s_tmin, s_tmax)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0 and is_rf_model(model):
            sigma_hat = sigma_hat.clamp(max=1.0 - 1e-4)

        if gamma > 0:
            if is_rf_model(model):
                x = rf_churn_step(x, sigmas[i], sigma_hat, eps)
            else:
                x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        dt = sigmas[i + 1] - sigma_hat

        # Euler method
        x = x + (math.cos(i + 1) / (i + 1) + 1) * d * dt
    return x
