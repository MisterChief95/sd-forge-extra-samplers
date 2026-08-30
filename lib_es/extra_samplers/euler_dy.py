import torch
from tqdm.auto import trange

from modules_forge.packages.k_diffusion.sampling import to_d

from lib_es.utils import (
    churn_gamma,
    dy_sampling_step,
    dy_sampling_step_blended,
    is_rf_model,
    sampler_metadata,
    substep_schedule,
)


@sampler_metadata("Euler Dy")
@torch.no_grad()
def sample_euler_dy(
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

    # Upstream RF control: keep its below-sigma-0.5 schedule and additive Dy substep.
    # EPS receives the same original indices but retains the stabilized blended update.
    dy_steps = set(substep_schedule(model, sigmas, (2, 3)))

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

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        outer_step = x + d * dt
        if rf:
            # Match upstream: advance the whole latent, then advance Dy's lattice by a
            # second dt while still evaluating it at sigma_hat.
            x_next = outer_step
            if sigmas[i + 1] > 0 and i in dy_steps:
                x_next = dy_sampling_step(x_next, model, dt, sigma_hat, **extra_args)
        else:
            if sigmas[i + 1] > 0 and i in dy_steps:
                x_next = dy_sampling_step_blended(x, outer_step, model, dt, sigma_hat, strength=0.5, **extra_args)
            else:
                x_next = outer_step

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        x = x_next

    return x
