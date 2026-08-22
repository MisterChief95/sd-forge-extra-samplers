import torch

from modules_forge.packages.k_diffusion.sampling import to_d

from tqdm.auto import trange

from lib_es.utils import churn_gamma, dy_sampling_step, is_rf_model, rf_churn_step, sampler_metadata


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

    # Half-resolution substep, always at the original step indices (2, 3) - deliberately
    # NOT relocated via substep_schedule for rectified flow. Running it near sigma ~1.0
    # lets it redefine composition (a DiT/RoPE model can reframe the scene, e.g. a
    # waist-up prompt coming back full-body) - that composition drift is a known, accepted
    # tradeoff here for the added generation variety it produces, not an oversight.
    dy_steps = {2, 3}

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = churn_gamma(s_churn, len(sigmas) - 1, sigmas[i], s_tmin, s_tmax)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0 and is_rf_model(model):
            sigma_hat = sigma_hat.clamp(max=1.0 - 1e-4)
        dt = sigmas[i + 1] - sigma_hat

        if i in dy_steps:
            x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)

        if gamma > 0:
            if is_rf_model(model):
                x = rf_churn_step(x, sigmas[i], sigma_hat, eps)
            else:
                x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Euler method
        x = x + d * dt

    return x
