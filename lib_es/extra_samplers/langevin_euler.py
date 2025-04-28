import torch
from tqdm.auto import trange
from k_diffusion.sampling import default_noise_sampler, to_d

import lib_es.const as consts
from lib_es.utils import sampler_metadata


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

    # Store original shape for aspect ratio calculations
    height, width = x.shape[2:4]
    aspect_ratio = width / height
    sigma_max = sigmas[0]

    langevin_strength = getattr(model.p, consts.LANGEVIN_STRENGTH, 0.1)

    for i in trange(len(sigmas) - 1, disable=disable):
        # Apply s_churn noise if requested
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
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
            # Simpler Langevin noise logic with less aggressive scaling
            # Use a constant base noise level with a gentle decay
            base_noise_level = langevin_strength  # Base level from parameter

            # Gentle decay curve - more consistent noise across steps
            # Sqrt provides a more gradual decrease than linear scaling
            decay_factor = torch.sqrt(sigmas[i + 1] / sigma_max)
            noise_scale = base_noise_level * (0.1 + 0.9 * decay_factor)

            # Higher safety clamp to allow more noise influence
            noise_scale = max(langevin_strength * 0.05, min(noise_scale, 0.8))

            # Generate balanced noise
            noise = torch.randn_like(x) * noise_scale
            height_scale = torch.sqrt(torch.tensor(aspect_ratio))
            width_scale = 1.0 / height_scale
            scaling = torch.tensor([1.0, 1.0, height_scale, width_scale]).reshape(1, -1, 1, 1).to(x.device)
            balanced_noise = noise * scaling

            x = x + balanced_noise

    return x
