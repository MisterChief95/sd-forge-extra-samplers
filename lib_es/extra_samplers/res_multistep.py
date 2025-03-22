import torch
from tqdm.auto import trange

from backend.modules.k_diffusion_extra import default_noise_sampler
from backend.patcher.unet import UnetPatcher
from k_diffusion.sampling import get_ancestral_step, to_d
from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion


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
    """
    Perform multi-step denoising using a conditional denoising model.
    Args:
        model (CFGDenoiserKDiffusion): The denoising model to use.
        x (torch.Tensor): The input tensor to be denoised.
        sigmas (list or torch.Tensor): A list or tensor of sigma values for each step.
        extra_args (dict, optional): Additional arguments to pass to the model. Defaults to None.
        callback (callable, optional): A callback function to be called after each step. Defaults to None.
        disable (bool, optional): If True, disables the progress bar. Defaults to None.
        s_churn (float, optional): Churn parameter for stochasticity. Defaults to 0.0.
        s_tmin (float, optional): Minimum sigma value for churn. Defaults to 0.0.
        s_tmax (float, optional): Maximum sigma value for churn. Defaults to float("inf").
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
    """
    Perform ancestral sampling with a multi-step method for denoising.
    Args:
        model (CFGDenoiserKDiffusion): The denoising model to use.
        x (torch.Tensor): The input tensor to be denoised.
        sigmas (list): A list of sigma values for the denoising process.
        extra_args (dict, optional): Additional arguments for the model. Defaults to None.
        callback (callable, optional): A callback function to be called after each step. Defaults to None.
        disable (bool, optional): If True, disables the progress bar. Defaults to None.
        s_churn (float, optional): Churn parameter for noise addition. Defaults to 0.0.
        s_tmin (float, optional): Minimum sigma value for churn. Defaults to 0.0.
        s_tmax (float, optional): Maximum sigma value for churn. Defaults to float("inf").
        eta (float, optional): Eta parameter for the ancestral step. Defaults to 1.0.
        s_noise (float, optional): Noise scale parameter. Defaults to 1.0.
        noise_sampler (callable, optional): Function to sample noise. Defaults to None.
        cfg_pp (bool, optional): If True, enables classifier-free guidance post-processing. Defaults to False.
    Returns:
        torch.Tensor: The denoised output tensor.
    """
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
    """Convenience function for sampling with the Res Multistep method."""
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
    """Convenience function for sampling with the Res Multistep method with CFG++."""
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
    """Convenience function for sampling with the Res Multistep Ancestral method."""
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
    """Convenience function for sampling with the Res Multistep Ancestral method with CFG++."""
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
