import torch
from tqdm import trange

from k_diffusion.sampling import get_ancestral_step, to_d

from lib_es.utils import default_noise_sampler, sampler_metadata


# ==============================================================================================================
#  - Originally written by aria1th: https://github.com/aria1th
#  - CFG++ support written by LaVie024: https://github.com/LaVie024
#  - Standard Euler support written by catboxanon: https://github.com/catboxanon
# ==============================================================================================================


@sampler_metadata(name="Euler a Multipass", extra_params={"uses_ensd": True})
@torch.no_grad()
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
    """
    A multipass variant of Euler-Ancestral sampling.
    - For each i in [0, len(sigmas)-2], we check if sigma_i is in [pass_sigma_min, pass_sigma_max].
      If so, subdivide the step from sigma_i -> sigma_{i+1} into 'pass_steps' sub-steps.
      Otherwise, do a single standard step.
    - Each sub-step calls 'get_ancestral_step(...)' with the sub-interval's start & end sigmas,
      then applies the usual Euler-Ancestral update:
         x = x + d*dt + (noise * sigma_up)
    """
    if extra_args is None:
        extra_args = {}

    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Decide how many sub-steps to do
        if pass_sigma_min <= sigma_i <= pass_sigma_max:
            n_sub = pass_steps
        else:
            n_sub = 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            # Current sub-step range:
            sub_sigma_curr = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            # Denoise at the current sub-sigma
            denoised = model(x, sub_sigma_curr * s_in, **extra_args)

            if callback is not None:
                callback({"x": x, "i": i, "sub_step": sub_step, "sigma": sub_sigma_curr, "denoised": denoised})

            # Compute the ancestral step parameters for this sub-interval
            sigma_down, sigma_up = get_ancestral_step(sub_sigma_curr, sub_sigma_next, eta=eta)
            if sigma_down == 0.0:
                # If we're stepping down to 0, we typically just take the final denoised
                x = denoised
            else:
                # Normal Euler-Ancestral logic
                d = to_d(x, sub_sigma_curr, denoised)
                dt = sigma_down - sub_sigma_curr
                x = x + d * dt
                if sigma_up != 0.0:
                    # Add noise for the "ancestral" part
                    x = x + noise_sampler(sub_sigma_curr, sub_sigma_next) * (s_noise * sigma_up)

    return x


@sampler_metadata(name="Euler Multipass")
@torch.no_grad()
def sample_euler_multipass(
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
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    """
    A multipass variant of Euler sampling.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Decide how many sub-steps to do
        if pass_sigma_min <= sigma_i <= pass_sigma_max:
            n_sub = pass_steps
        else:
            n_sub = 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            # Current sub-step range:
            sub_sigma_curr = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            if s_churn > 0:
                gamma = min(s_churn / (n_sub - 1), 2**0.5 - 1) if s_tmin <= sub_sigma_curr < s_tmax else 0
                sigma_hat = sub_sigma_curr * (gamma + 1)
            else:
                gamma = 0
                sigma_hat = sub_sigma_curr

            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + eps * (sigma_hat**2 - sigma_hat**2) ** 0.5

            # Denoise at the current sub-sigma
            denoised = model(x, sub_sigma_curr * s_in, **extra_args)

            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sub_step": sub_step,
                        "sigma": sub_sigma_curr,
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    }
                )

            d = to_d(x, sigma_hat, denoised)
            dt = sub_sigma_next - sigma_hat
            # Euler method
            x = x + d * dt
    return x


@sampler_metadata(name="Euler Multipass CFG++")
@torch.no_grad()
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
    """
    CFG++-enabled multipass Euler sampler.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        n_sub = pass_steps if pass_sigma_min <= sigma_i <= pass_sigma_max else 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            sub_sigma = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            if s_churn > 0:
                gamma = min(s_churn / (n_sub - 1), 2**0.5 - 1) if s_tmin <= sub_sigma < s_tmax else 0
                sigma_hat = sub_sigma * (gamma + 1)
            else:
                gamma = 0
                sigma_hat = sub_sigma

            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + eps * (sigma_hat**2 - sub_sigma**2).sqrt()

            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = model.last_noise_uncond

            x = denoised + d * sub_sigma_next

            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

    return x


@sampler_metadata(name="Euler a Multipass CFG++", extra_params={"uses_ensd": True})
@torch.no_grad()
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
    """
    CFG++-enabled multipass ancestral Euler sampler.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Subdivision
        n_sub = pass_steps if pass_sigma_min <= sigma_i <= pass_sigma_max else 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            sub_sigma = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            denoised = model(x, sub_sigma * s_in, **extra_args)
            d = model.last_noise_uncond

            # Compute ancestral steps
            sigma_down, sigma_up = get_ancestral_step(sub_sigma, sub_sigma_next, eta=eta)

            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sub_step": sub_step,
                        "sigma": sub_sigma,
                        "sigma_hat": sub_sigma,
                    }
                )

            # Main ancestral Euler update with CFG++
            x = denoised + d * sigma_down

            # Noise injection
            if sub_sigma_next > 0:
                x = x + noise_sampler(sub_sigma, sub_sigma_next) * s_noise * sigma_up
    return x
