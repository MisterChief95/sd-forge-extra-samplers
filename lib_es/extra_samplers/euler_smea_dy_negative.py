import torch

from modules_forge.packages.k_diffusion.sampling import to_d

from tqdm.auto import trange

from lib_es.utils import dy_sampling_step, smea_sampling_step
from lib_es.utils import churn_gamma, is_rf_model, rf_churn_step, sampler_metadata


@sampler_metadata("Euler SMEA Dy Negative")
@torch.no_grad()
def sample_euler_smea_dy_negative(
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

    # NOTE: deliberately NOT on substep_schedule, unlike the other SMEA/Dy samplers.
    #
    # Here each substep is followed by `x = -x - d * dt`, and that negation is only
    # survivable while the latent is still mostly noise. Moving the substeps below
    # sigma 0.5 - which fixes composition drift everywhere else - instead negates a
    # nearly-final latent and comes back with an inverted palette. The resolution
    # substeps and the sign flip cannot be rescheduled independently, so this sampler
    # keeps its original high-sigma schedule until the negation itself is revisited.
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = churn_gamma(s_churn, len(sigmas) - 1, sigmas[i], s_tmin, s_tmax)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0 and is_rf_model(model):
            sigma_hat = sigma_hat.clamp(max=1.0 - 1e-4)
        dt = sigmas[i + 1] - sigma_hat

        if gamma > 0:
            if is_rf_model(model):
                x = rf_churn_step(x, sigmas[i], sigma_hat, eps)
            else:
                x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        # Euler method
        x = x + d * dt

        if sigmas[i + 1] > 0:
            if (i + 1) // 2 == 1:
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
                x = -x - d * dt

            if (i + 1) // 2 == 0:
                x = smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
                x = -x - d * dt

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

    return x
