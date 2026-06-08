import torch
from tqdm.auto import trange
from modules_forge.packages.k_diffusion.sampling import (
    BrownianTreeNoiseSampler,
    get_ancestral_step,
    to_d,
)

from lib_es.utils import sampler_metadata


@sampler_metadata(
    "Heun Ancestral",
    {"uses_ensd": True, "sampler_order": 2},
)
@torch.no_grad()
def sample_heun_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """
    Ancestral sampling with Heun's method steps.
    Based on Algorithm 2 (Heun steps) from Karras et al. (2022).

    Args:
        model: The model to sample from.
        x: The initial noise.
        sigmas: The noise levels to sample at.
        extra_args: Extra arguments to the model.
        callback: A function that's called after each step.
        disable: Disable tqdm progress bar.
        eta: Ancestral sampling strength parameter.
        s_noise: Noise scale.
        noise_sampler: A function that returns noise.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = (
        BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
        if noise_sampler is None
        else noise_sampler
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        dt = sigmas[i + 1] - sigmas[i]

        if sigmas[i + 1] == 0:
            # Euler method for final step
            x = x + d * dt
        else:
            # Heun's method (predictor-corrector)
            # 1. Predictor step (Euler)
            x_2 = x + d * dt

            # 2. Evaluate at the predicted point (at sigmas[i+1], not sigma_down)
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)

            # 3. Corrector step (average of derivatives)
            d_prime = (d + d_2) / 2

            # Calculate ancestral step parameters
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

            # Step to sigma_down using averaged derivative
            dt_down = sigma_down - sigmas[i]
            x = x + d_prime * dt_down

            # Add noise to reach sigmas[i+1]
            if sigma_up > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

    return x
