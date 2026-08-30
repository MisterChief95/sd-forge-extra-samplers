import torch

from modules_forge.packages.k_diffusion.sampling import to_d

from tqdm.auto import trange

from lib_es.utils import dy_sampling_step, dy_sampling_step_blended, smea_sampling_step
from lib_es.utils import churn_gamma, is_rf_model, sampler_metadata


@sampler_metadata("Euler SMEA Dy")
@torch.no_grad()
def sample_euler_smea_dy(
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

    # Upstream RF control: SMEA at step 0 and additive Dy at steps 1 and 2, each after
    # the outer Euler update. EPS stays on the stabilized single-Dy blended schedule.
    smea_steps = {0}
    dy_steps = {1, 2} if rf else {1}

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

        # RF matches upstream's additive ordering. EPS keeps its one-dt blended update.
        outer_step = x + d * dt
        if rf:
            x = outer_step
            if sigmas[i + 1] > 0 and i in dy_steps:
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
            if sigmas[i + 1] > 0 and i in smea_steps:
                x = smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
        elif sigmas[i + 1] > 0 and i in dy_steps:
            x = dy_sampling_step_blended(x, outer_step, model, dt, sigma_hat, strength=0.5, **extra_args)
        elif sigmas[i + 1] > 0 and i in smea_steps:
            smea_step = smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
            x = torch.lerp(outer_step, smea_step, 0.5)
        else:
            x = outer_step

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

    return x
