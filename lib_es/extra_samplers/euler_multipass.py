import torch
from tqdm import trange

from modules_forge.packages.k_diffusion.sampling import (
    offset_first_sigma_for_snr,
    sample_euler_ancestral_RF,
    to_d,
)

from lib_es.utils import (
    ancestral_step,
    scale_sigma_threshold,
    cfg_pp_ancestral_step,
    default_noise_sampler,
    extend_sigmas,
    sampler_metadata,
    setup_cfg_pp,
    is_rf_model,
    rf_churn_step,
)


# ==============================================================================================================
#  - Originally written by aria1th: https://github.com/aria1th
#  - CFG++ support written by LaVie024: https://github.com/LaVie024
#  - Standard Euler support written by catboxanon: https://github.com/catboxanon
# ==============================================================================================================


def apply_churn(model, x, sub_sigma, s_churn, s_tmin, s_tmax, s_noise, pass_step):
    if s_churn > 0:
        gamma = min(s_churn / max(0, pass_step - 1), 2**0.5 - 1) if s_tmin <= sub_sigma < s_tmax else 0
        sigma_hat = sub_sigma * (gamma + 1)
    else:
        gamma = 0
        sigma_hat = sub_sigma

    if gamma > 0 and is_rf_model(model):
        sigma_hat = sigma_hat.clamp(max=1.0 - 1e-4)

    if gamma > 0:
        eps = torch.randn_like(x) * s_noise
        if is_rf_model(model):
            x = rf_churn_step(x, sub_sigma, sigma_hat, eps)
        else:
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
    - Each sub-step calls 'ancestral_step(...)' with the sub-interval's start & end sigmas,
      then applies the usual Euler-Ancestral update:
         x = x + d*dt + (noise * sigma_up)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_sampling = model.inner_model.predictor

    if cfg_pp:
        extra_args = setup_cfg_pp(extra_args, post_cfg_function)
        # alpha_s is 0 at sigma=1, where rectified-flow schedules start; no-op on eps.
        sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    # pass_sigma_min defaults to 12.0, chosen for eps schedules that run to ~14.6 so that
    # only the first handful of high-noise steps get subdivided. Rectified-flow sigma never
    # exceeds 1.0, so the window [pass_sigma_min, pass_sigma_max] excluded the entire
    # schedule and the multipass subdivision silently did nothing at all. Rescaling keeps
    # the same early-step window on both families; eps is unchanged.
    sub_sigmas = extend_sigmas(sigmas, pass_steps, pass_sigma_max, scale_sigma_threshold(model, pass_sigma_min, sigmas))

    if not cfg_pp and is_rf_model(model):
        return sample_euler_ancestral_RF(
            model, x, sub_sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler
        )

    for i in trange(len(sub_sigmas) - 1, disable=disable):
        # Current sub-step range:
        sub_sigma_curr = sub_sigmas[i]
        sub_sigma_next = sub_sigmas[i + 1]

        # Denoise at the current sub-sigma
        denoised = model(x, sub_sigma_curr * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sub_step": i, "sigma": sub_sigma_curr, "denoised": denoised})

        if cfg_pp:
            # CFG++ carries its own alpha-aware ancestral split, so it does not go through
            # ancestral_step / alpha_ratio.
            x, alpha_t, sigma_up = cfg_pp_ancestral_step(
                model_sampling, x, sub_sigma_curr, sub_sigma_next, denoised, uncond_denoised, eta
            )
            if sigma_up != 0.0:
                x = x + alpha_t * noise_sampler(sub_sigma_curr, sub_sigma_next) * (s_noise * sigma_up)
            continue

        # Compute the ancestral step parameters for this sub-interval
        sigma_down, sigma_up, alpha_ratio = ancestral_step(model, sub_sigma_curr, sub_sigma_next, eta)

        d = to_d(x, sub_sigma_curr, denoised)

        if sigma_down == 0.0:
            x = denoised
        else:
            x = x + d * (sigma_down - sub_sigma_curr)

        # Rescale the signal (rectified flow only) before the ancestral renoise
        if alpha_ratio is not None:
            x = alpha_ratio * x

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

    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_sampling = model.inner_model.predictor

    if cfg_pp:
        extra_args = setup_cfg_pp(extra_args, post_cfg_function)
        # alpha_s is 0 at sigma=1, where rectified-flow schedules start; no-op on eps.
        sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    s_in = x.new_ones([x.shape[0]])
    # pass_sigma_min defaults to 12.0, chosen for eps schedules that run to ~14.6 so that
    # only the first handful of high-noise steps get subdivided. Rectified-flow sigma never
    # exceeds 1.0, so the window [pass_sigma_min, pass_sigma_max] excluded the entire
    # schedule and the multipass subdivision silently did nothing at all. Rescaling keeps
    # the same early-step window on both families; eps is unchanged.
    sub_sigmas = extend_sigmas(sigmas, pass_steps, pass_sigma_max, scale_sigma_threshold(model, pass_sigma_min, sigmas))

    for i in trange(len(sub_sigmas) - 1, disable=disable):
        # Current sub-step range:
        sub_sigma_curr = sub_sigmas[i]
        sub_sigma_next = sub_sigmas[i + 1]

        x, sigma_hat = apply_churn(model, x, sub_sigma_curr, s_churn, s_tmin, s_tmax, s_noise, pass_steps)

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

        if cfg_pp:
            # Deterministic CFG++ step (eta=0); alpha-aware, so it works on RF too.
            x, _, _ = cfg_pp_ancestral_step(
                model_sampling, x, sigma_hat, sub_sigma_next, denoised, uncond_denoised, eta=0.0
            )
        else:
            d = to_d(x, sigma_hat, denoised)
            x = x + d * (sub_sigma_next - sigma_hat)

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
