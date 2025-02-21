import torch
from tqdm.auto import trange

from backend.modules.k_diffusion_extra import default_noise_sampler
from backend.patcher.unet import UnetPatcher
from backend.sampling.condition import ConditionCrossAttn
from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion
from k_diffusion.sampling import get_ancestral_step, to_d


def sigma_fn(t):
    return t.neg().exp()


def t_fn(sigma):
    return sigma.log().neg()


def phi1_fn(t):
    return torch.expm1(t) / t


def phi2_fn(t):
    return (phi1_fn(t) - 1.0) / t


def construct_empty_uncond(cond):
    dict(
        cross_attn=cond,
        model_conds=dict(
            c_crossattn=ConditionCrossAttn(cond),
        ),
    )


@torch.no_grad()
def res_multistep(
    model: CFGDenoiserKDiffusion,
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
    cfg_pp=False,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model.need_last_noise_uncond = True
        unet_patcher: UnetPatcher = model.inner_model.inner_model.forge_objects.unet
        unet_patcher.model_options["disable_cfg1_optimization"] = True
        unet_patcher.set_model_sampler_post_cfg_function(post_cfg_function)

    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        if sigmas[i + 1] == 0 or old_denoised is None:
            # Euler method
            if cfg_pp:
                d = model.last_noise_uncond
                x = denoised + d * sigmas[i + 1]
            else:
                d = to_d(x, sigma_hat, denoised)
                dt = sigmas[i + 1] - sigma_hat
                x = x + d * dt
        else:
            # Second order multistep method in https://arxiv.org/pdf/2308.02157
            t, t_next, t_prev = t_fn(sigmas[i]), t_fn(sigmas[i + 1]), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
            b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

            if cfg_pp:
                # uncond_denoised = x - model.last_noise_uncond * sigmas[i] - alternative to post CFG function
                x = x + (denoised - uncond_denoised)

            x = (sigma_fn(t_next) / sigma_fn(t)) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
    return x


@torch.no_grad()
def res_multistep_ancestral(
    model: CFGDenoiserKDiffusion,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    cfg_pp=False,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model.need_last_noise_uncond = True
        unet_patcher: UnetPatcher = model.inner_model.inner_model.forge_objects.unet
        unet_patcher.model_options["disable_cfg1_optimization"] = True
        unet_patcher.set_model_sampler_post_cfg_function(post_cfg_function)

    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Calculate sigma_down and sigma_up for the ancestral step
        sigma_down, sigma_up = get_ancestral_step(sigma_hat, sigmas[i + 1], eta=eta)

        if sigmas[i + 1] == 0 or old_denoised is None:
            # Euler method for the first step
            if cfg_pp:
                d = model.last_noise_uncond
                x = denoised + d * sigmas[i + 1]
            else:
                d = to_d(x, sigma_hat, denoised)
                dt = sigma_down - sigma_hat
                x = x + d * dt
        else:
            # Second-order multistep method
            t, t_next, t_prev = t_fn(sigma_hat), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
            b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

            x = (sigma_fn(t_next) / sigma_fn(t)) * x + h * (b1 * denoised + b2 * old_denoised)

            if cfg_pp:
                # uncond_denoised = x - model.last_noise_uncond * sigmas[i] - alternative to post CFG function
                x = x + (denoised - uncond_denoised)

        # Ancestral noise addition
        if sigma_up > 0:
            noise = noise_sampler(sigma_hat, sigmas[i + 1])
            x = x + noise * s_noise * sigma_up

        old_denoised = denoised

    return x


@torch.no_grad()
def sample_res_multistep(
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
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_churn=s_churn,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        cfg_pp=False,
    )


@torch.no_grad()
def sample_res_multistep_cfgpp(
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
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_churn=s_churn,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        cfg_pp=True,
    )


@torch.no_grad()
def sample_res_multistep_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    noise_sampler=None,
):
    return res_multistep_ancestral(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_churn=s_churn,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        cfg_pp=False,
    )


@torch.no_grad()
def sample_res_multistep_ancestral_cfgpp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    noise_sampler=None,
):
    return res_multistep_ancestral(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_churn=s_churn,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        cfg_pp=True,
    )
