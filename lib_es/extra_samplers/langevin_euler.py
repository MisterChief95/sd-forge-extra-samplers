import torch
from tqdm.auto import trange
from modules_forge.packages.k_diffusion.sampling import default_noise_sampler, to_d

import lib_es.const as consts
from lib_es.utils import is_rf_model, rf_churn_step, sampler_metadata


@sampler_metadata(
    "Langevin Euler",
    {"scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_langevin_euler(
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
    noise_sampler=None,
):
    """
    Langevin dynamics sampler - the adaptive CFG is now handled by the CFG function.
    This is your original implementation but with the adaptive CFG logic removed.
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    sigma_max = sigmas[0]

    langevin_strength = getattr(model.p, consts.LANGEVIN_STRENGTH, 0.1)

    for i in trange(len(sigmas) - 1, disable=disable):
        # Apply s_churn noise if requested
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0 and is_rf_model(model):
            sigma_hat = sigma_hat.clamp(max=1.0 - 1e-4)
        if gamma > 0:
            if is_rf_model(model):
                x = rf_churn_step(x, sigmas[i], sigma_hat, eps)
            else:
                x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        # Perform model prediction - CFG is now handled by our function
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Call the callback
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Calculate the derivative
        d = to_d(x, sigma_hat, denoised)

        # Langevin step: Deterministic part + Noise part
        dt = sigmas[i + 1] - sigma_hat

        # Deterministic Euler step
        x = x + d * dt

        # Apply Langevin noise if not the final step
        if sigmas[i + 1] > 0:
            sigma_delta = (sigma_hat - sigmas[i + 1]).abs()
            decay_factor = (sigmas[i + 1] / sigma_max).clamp(min=0).sqrt()
            noise_scale = langevin_strength * sigma_delta * decay_factor
            noise_scale = torch.minimum(noise_scale, sigmas[i + 1] * 0.5)

            noise = noise_sampler(sigmas[i], sigmas[i + 1])

            if is_rf_model(model):
                # The injection raises the effective noise level above sigma_next, so RF
                # needs the signal scaled down to match or the latent drifts off the
                # (1 - t) * x0 + t * n manifold. Expressing it as a churn from sigma_next
                # up to the combined level applies exactly that correction.
                sigma_eff = (sigmas[i + 1] ** 2 + noise_scale**2).sqrt()
                x = rf_churn_step(x, sigmas[i + 1], sigma_eff, noise)
            else:
                x = x + noise * noise_scale

    return x
