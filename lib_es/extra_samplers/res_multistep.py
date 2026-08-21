import torch
from functools import partial
from tqdm.auto import trange

from modules_forge.packages.k_diffusion.sampling import (
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
    to_d,
    default_noise_sampler,
)

from lib_es.utils import (
    is_rf_model,
    alpha_for,
    ancestral_step,
    cfg_pp_ancestral_step,
    cfg_pp_noise_params,
    sampler_metadata,
    setup_cfg_pp,
)

# ==============================================================================================
# Second order multistep solver from https://arxiv.org/pdf/2308.02157
#
# Adapted from ComfyUI's comfy/k_diffusion/sampling.py `res_multistep`:
#   https://github.com/comfyanonymous/ComfyUI  (GPL-3.0)
#
# Deviates from upstream in one respect: upstream is eps-only. It takes the exponential
# integrator's time variable as -log(sigma), which is the half-logSNR only when alpha == 1,
# and splits ancestral steps with the alpha==1 get_ancestral_step. Rectified-flow models have
# alpha = 1 - sigma, so both are wrong for them. Here the time variable comes from the model's
# own predictor via sigma_to_half_log_snr, and the ancestral split is dispatched per model
# family. Both reduce to the upstream expressions exactly when alpha == 1, so eps-model output
# is unchanged.
# ==============================================================================================


def phi1_fn(t):
    """
    Computes the function phi1(t) = (exp(t) - 1) / t using PyTorch's expm1 function.
    Args:
        t (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: The result of (exp(t) - 1) / t.
    """

    return torch.expm1(t) / t


def phi2_fn(t):
    """
    Compute the value of the phi2 function.
    The phi2 function is defined as (phi1_fn(t) - 1.0) / t, where phi1_fn is
    another function that takes a single argument t.
    Parameters:
    t (float): The input value for the function.
    Returns:
    float: The computed value of the phi2 function.
    """

    return (phi1_fn(t) - 1.0) / t


@torch.no_grad()
def res_multistep(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    eta=1.0,
    cfg_pp=False,
):
    """
    Perform multi-step denoising using a conditional denoising model.
    Args:
        model (CFGDenoiserKDiffusion): The denoising model to use.
        x (torch.Tensor): The input tensor to be denoised.
        sigmas (list or torch.Tensor): A list or tensor of sigma values for each step.
        extra_args (dict, optional): Additional arguments to pass to the model. Defaults to None.
        callback (callable, optional): A callback function to be called after each step. Defaults to None.
        disable (bool, optional): If True, disables the progress bar. Defaults to None.
        s_noise (float, optional): Noise scale for stochasticity. Defaults to 1.0.
        noise_sampler (callable, optional): Function to sample noise. Defaults to None.
        cfg_pp (bool, optional): If True, enables post-processing for classifier-free guidance. Defaults to False.
    Returns:
        torch.Tensor: The denoised output tensor.
    """
    extra_args = {} if extra_args is None else extra_args

    if cfg_pp and is_rf_model(model):
        raise RuntimeError(
            "Res Multistep CFG++ is not supported on rectified-flow models "
            "(Flux, SD3, Qwen-Image, Krea, Anima, Wan).\n"
            "\n"
            "This sampler's second-order CFG++ step has no published rectified-flow form, "
            "and the generalizations tried here produced pure noise rather than an image. "
            "Rather than emit a broken result it stops here.\n"
            "\n"
            "Use one of these instead, which are verified on rectified flow:\n"
            "  - Euler a CFG++            (built in)\n"
            "  - Euler a Multipass CFG++\n"
            "  - Gradient Estimation CFG++\n"
            "  - Res Multistep Ancestral  (no CFG++; second-order and rectified-flow correct)\n"
            "\n"
            "Res Multistep CFG++ remains fully supported on epsilon models (SD1.5, SDXL)."
        )

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    old_sigma_down = None
    old_denoised = None
    uncond_denoised = None

    # unconditional denoised is used for the second order multistep method
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        extra_args = setup_cfg_pp(extra_args, post_cfg_function)

    # The exponential integrator below runs in half-logSNR space, lambda = log(alpha/sigma).
    # This used to hardcode -log(sigma), which is only lambda when alpha == 1; rectified-flow
    # models have alpha = 1 - sigma, so lambda is logit(sigma).neg() instead. Deriving it via
    # the model's predictor keeps both families correct - the same approach seeds_2/seeds_3
    # and sa_solver already take. offset_first_sigma_for_snr nudges a leading sigma of exactly
    # 1.0 off the boundary, where RF's logit would otherwise be infinite.
    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up, alpha_ratio = ancestral_step(model, sigmas[i], sigmas[i + 1], eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigma_down == 0 or old_denoised is None:
            # Euler method
            if cfg_pp:
                d = to_d(x, sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                d = to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
        else:
            # Second order multistep method in https://arxiv.org/pdf/2308.02157
            lambda_s, lambda_old = lambda_fn(sigmas[i]), lambda_fn(old_sigma_down)
            lambda_next, lambda_prev = lambda_fn(sigma_down), lambda_fn(sigmas[i - 1])
            h = lambda_next - lambda_s
            c2 = (lambda_prev - lambda_old) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            # DPM-Solver++ data prediction: x carries at sigma_down/sigma_s, and the data
            # terms are weighted by alpha at the destination. For alpha == 1 this reduces to
            # the previous exp(-h) * x + h * (...) form exactly.
            sigma_ratio = sigma_down / sigmas[i]
            alpha_down = alpha_for(model_sampling, sigma_down)

            # Only eps reaches this with cfg_pp set; RF took the first-order branch above.
            if cfg_pp:
                x = x + (denoised - uncond_denoised)
                x = sigma_ratio * x + alpha_down * h * (b1 * uncond_denoised + b2 * old_denoised)
            else:
                x = sigma_ratio * x + alpha_down * h * (b1 * denoised + b2 * old_denoised)

        # Rescale the signal (rectified flow only), then add the ancestral noise
        if alpha_ratio is not None:
            x = alpha_ratio * x

        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        old_denoised = uncond_denoised if cfg_pp else denoised
        old_sigma_down = sigma_down
    return x


@sampler_metadata(
    "Res Multistep",
    {"scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep(
    model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=0.0,
        cfg_pp=False,
    )


@sampler_metadata(
    "Res Multistep CFG++",
    {"scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep_cfg_pp(
    model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=0.0,
        cfg_pp=True,
    )


@sampler_metadata(
    "Res Multistep Ancestral",
    {"uses_ensd": True, "scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep_ancestral(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=eta,
        cfg_pp=False,
    )


@sampler_metadata(
    "Res Multistep Ancestral CFG++",
    {"uses_ensd": True, "scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep_ancestral_cfg_pp(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=eta,
        cfg_pp=True,
    )
