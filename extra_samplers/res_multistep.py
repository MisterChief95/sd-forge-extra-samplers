import torch
from tqdm.auto import trange

from backend.modules.k_diffusion_extra import default_noise_sampler
from backend.patcher.base import set_model_options_post_cfg_function
from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion
from k_diffusion.sampling import to_d


@torch.no_grad()
def res_multistep(model: CFGDenoiserKDiffusion, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None, cfg_pp=False):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

    old_denoised = None
    uncond_denoised = None

    # Unable to get uncond_denoised in forge?
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model.need_last_noise_uncond = True
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)


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
                x = x + (denoised - uncond_denoised)

            x = (sigma_fn(t_next) / sigma_fn(t)) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, noise_sampler=noise_sampler, cfg_pp=False)

@torch.no_grad()
def sample_res_multistep_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, noise_sampler=noise_sampler, cfg_pp=True)