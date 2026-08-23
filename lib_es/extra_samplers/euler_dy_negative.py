import torch

from modules_forge.packages.k_diffusion.sampling import to_d

from tqdm.auto import trange

from lib_es.utils import churn_gamma, dy_sampling_step, is_rf_model, sampler_metadata


@sampler_metadata("Euler Dy Negative")
@torch.no_grad()
def sample_euler_dy_negative(
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

    # NOTE: deliberately NOT on substep_schedule, unlike euler_dy.
    #
    # On the substep this replaces the Euler update with `x = -x - d * dt`, which is only
    # survivable while the latent is still mostly noise. Rescheduling it below sigma 0.5 -
    # the fix applied to the non-negative SMEA/Dy samplers - negates a nearly-final latent
    # and returns an inverted palette instead of an image. Same coupling as
    # euler_smea_dy_negative: the substep and the sign flip cannot move independently.
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

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Euler method
        if sigmas[i + 1] > 0 and i // 2 == 1:
            x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
            x = -x - d * dt
        else:
            x = x + d * dt

    return x
