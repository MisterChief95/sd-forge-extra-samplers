import torch

from modules_forge.packages.k_diffusion.sampling import to_d

from tqdm.auto import trange

from lib_es.utils import (
    churn_gamma,
    dy_sampling_step,
    is_rf_model,
    sampler_metadata,
    smea_sampling_step,
)


@sampler_metadata("Euler SMEA")
@torch.no_grad()
def sample_euler_smea(
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
    rf = is_rf_model(model)

    # Upstream RF control: its standalone sampler is currently a pre-Euler Dy lattice
    # operation at steps 2 and 3 despite the SMEA label. EPS remains genuine 1.25x SMEA.
    rf_dy_steps = {2, 3}
    smea_steps = {0}

    for i in trange(len(sigmas) - 1, disable=disable):
        if rf:
            gamma = 0.0
        else:
            gamma = churn_gamma(model, s_churn, len(sigmas) - 1, sigmas[i], s_tmin, s_tmax)
            eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat

        if gamma > 0:
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        if rf and i in rf_dy_steps:
            x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        outer_step = x + d * dt
        if not rf and sigmas[i + 1] > 0 and i in smea_steps:
            # Preserve the complete enlarged-resolution EPS Euler target.
            x_next = smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
        else:
            x_next = outer_step

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Euler method
        x = x_next

    return x
