import torch
from tqdm import trange

from modules_forge.packages.k_diffusion.sampling import get_ancestral_step, to_d

from lib_es.utils import default_noise_sampler, extend_sigmas, sampler_metadata


# ==============================================================================================================
#  - Originally written by aria1th: https://github.com/aria1th
#  - CFG++ support written by LaVie024: https://github.com/LaVie024
#  - Standard Euler support written by catboxanon: https://github.com/catboxanon
# ==============================================================================================================


def apply_churn(x, sub_sigma, s_churn, s_tmin, s_tmax, s_noise, pass_step):
    if s_churn > 0:
        gamma = min(s_churn / max(0, pass_step - 1), 2**0.5 - 1) if s_tmin <= sub_sigma < s_tmax else 0
        sigma_hat = sub_sigma * (gamma + 1)
    else:
        gamma = 0
        sigma_hat = sub_sigma

    if gamma > 0:
        eps = torch.randn_like(x) * s_noise
        x = x + eps * (sigma_hat**2 - sub_sigma**2) ** 0.5

    return x, sigma_hat


@torch.no_grad()
def euler_ancestral_multipass(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
    cfg_pp=False,
):
    """
    A multipass variant of Euler-Ancestral sampling.
    - For each i in [0, len(sigmas)-2], we check if sigma_i is in [pass_sigma_min, pass_sigma_max].
      If so, subdivide the step from sigma_i -> sigma_{i+1} into 'pass_steps' sub-steps.
      Otherwise, do a single standard step.
    - Each sub-step calls 'get_ancestral_step(...)' with the sub-interval's start & end sigmas,
      then applies the usual Euler-Ancestral update:
         x = x + d*dt + (noise * sigma_up)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    if cfg_pp:
        model.need_last_noise_uncond = True
        model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    sub_sigmas = extend_sigmas(sigmas, pass_steps, pass_sigma_max, pass_sigma_min)

    for i in trange(len(sub_sigmas) - 1, disable=disable):
        # Current sub-step range:
        sub_sigma_curr = sub_sigmas[i]
        sub_sigma_next = sub_sigmas[i + 1]

        # Denoise at the current sub-sigma
        denoised = model(x, sub_sigma_curr * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sub_step": i, "sigma": sub_sigma_curr, "denoised": denoised})

        # Compute the ancestral step parameters for this sub-interval
        sigma_down, sigma_up = get_ancestral_step(sub_sigma_curr, sub_sigma_next, eta=eta)

        d = model.last_noise_uncond if cfg_pp else to_d(x, sub_sigma_curr, denoised)

        if cfg_pp:
            x = denoised + d * sigma_down
        elif sigma_down == 0.0:
            x = denoised
        else:
            x = x + d * (sigma_down - sub_sigma_curr)

        if sigma_up != 0.0:
            # Add noise for the "ancestral" part
            x = x + noise_sampler(sub_sigma_curr, sub_sigma_next) * (s_noise * sigma_up)

    return x


@sampler_metadata(name="Euler a Multipass", extra_params={"uses_ensd": True})
def sample_euler_ancestral_multipass(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    return euler_ancestral_multipass(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        eta,
        s_noise,
        noise_sampler,
        pass_steps,
        pass_sigma_max,
        pass_sigma_min,
        False,
    )


@sampler_metadata(name="Euler a Multipass CFG++", extra_params={"uses_ensd": True})
def sample_euler_ancestral_multipass_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    return euler_ancestral_multipass(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        eta,
        s_noise,
        noise_sampler,
        pass_steps,
        pass_sigma_max,
        pass_sigma_min,
        True,
    )


@torch.no_grad()
def euler_multipass(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
    cfg_pp=False,
):
    """
    A multipass variant of Euler sampling.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    if cfg_pp:
        model.need_last_noise_uncond = True
        model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    s_in = x.new_ones([x.shape[0]])
    sub_sigmas = extend_sigmas(sigmas, pass_steps, pass_sigma_max, pass_sigma_min)

    for i in trange(len(sub_sigmas) - 1, disable=disable):
        # Current sub-step range:
        sub_sigma_curr = sub_sigmas[i]
        sub_sigma_next = sub_sigmas[i + 1]

        x, sigma_hat = apply_churn(x, sub_sigma_curr, s_churn, s_tmin, s_tmax, s_noise, pass_steps)

        # Denoise at the current sub-sigma
        denoised = model(x, sub_sigma_curr * s_in, **extra_args)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sub_step": i,
                    "sigma": sub_sigma_curr,
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )

        d = model.last_noise_uncond if cfg_pp else to_d(x, sigma_hat, denoised)
        x = denoised + d * sub_sigma_next if cfg_pp else x + d * (sub_sigma_next - sigma_hat)

    return x


@sampler_metadata(name="Euler Multipass")
def sample_euler_multipass(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    return euler_multipass(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        noise_sampler,
        s_churn,
        s_tmin,
        s_tmax,
        s_noise,
        pass_steps,
        pass_sigma_max,
        pass_sigma_min,
        False,
    )


@sampler_metadata(name="Euler Multipass CFG++")
def sample_euler_multipass_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    return euler_multipass(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        noise_sampler,
        s_churn,
        s_tmin,
        s_tmax,
        s_noise,
        pass_steps,
        pass_sigma_max,
        pass_sigma_min,
        True,
    )
