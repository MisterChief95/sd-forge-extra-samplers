import torch
from tqdm.auto import trange

from modules_forge.packages.k_diffusion.sampling import get_ancestral_step, to_d, default_noise_sampler
from backend.patcher.unet import UnetPatcher

from lib_es.utils import sampler_metadata


def sigma_fn(t):
    """
    Computes the sigma function for a given tensor `t`.
    The sigma function is defined as the exponential of the negation of `t`.
    Args:
        t (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: The result of applying the sigma function to `t`.
    """

    return t.neg().exp()


def t_fn(sigma):
    """
    Computes the negative logarithm of the input tensor.
    Args:
        sigma (torch.Tensor): A tensor for which the negative logarithm is to be computed.
    Returns:
        torch.Tensor: A tensor containing the negative logarithm of the input tensor.
    """

    return sigma.log().neg()


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
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    uncond_denoised = None

    # unconditional denoised is used for the second order multistep method
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model.need_last_noise_uncond = True
        unet_patcher: UnetPatcher = model.inner_model.inner_model.forge_objects.unet
        unet_patcher.model_options["disable_cfg1_optimization"] = True  # not sure if this really works
        unet_patcher.set_model_sampler_post_cfg_function(post_cfg_function)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
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
            t, t_next, t_prev = t_fn(sigmas[i]), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            if cfg_pp:
                x = x + (denoised - uncond_denoised)
                x = sigma_fn(h) * x + h * (b1 * uncond_denoised + b2 * old_denoised)
            else:
                x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)

        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if cfg_pp:
            old_denoised = uncond_denoised
        else:
            old_denoised = denoised
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
