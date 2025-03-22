import torch
from tqdm.auto import trange
from k_diffusion.sampling import to_d


def default_noise_sampler(x):
    """Returns a function that generates random normal noise of the same shape as x."""
    return lambda sigma, sigma_next: torch.randn_like(x)


@torch.no_grad()
def sample_langevin_plms(
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
    langevin_strength=0.1,
    cfg_alpha=0.3,
    cfg_beta=2.0,
    cfg_gamma=0.1,
    plms_order=4,
):
    """
    Langevin dynamics sampler with PLMS and adaptive CFG.

    Args:
        model: The model to sample from.
        x: The initial noise.
        sigmas: The noise levels to sample at.
        extra_args: Extra arguments to the model.
        callback: A function that's called after each step.
        disable: Disable tqdm progress bar.
        s_churn: Amount of noise to add per step.
        s_tmin: Minimum sigma for adding noise.
        s_tmax: Maximum sigma for adding noise.
        s_noise: Noise scale.
        noise_sampler: A function that returns noise.
        langevin_strength: Strength of the Langevin noise term.
        cfg_alpha: Controls magnitude of CFG increase during sampling.
        cfg_beta: Controls how quickly CFG changes occur.
        cfg_gamma: Controls reduction of CFG at very low noise levels.
        plms_order: Order of the PLMS method (max 4)
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # Store original cfg scale
    original_cfg = getattr(model.p, "cfg_scale", 7.0)
    sigma_max = sigmas[0]

    # Store original shape for aspect ratio calculations
    height, width = x.shape[2:4]
    aspect_ratio = width / height

    # Initialize for CFG smoothing
    last_cfg_multiplier = 1.0

    # Storage for PLMS
    ds = []  # Store denoising terms
    plms_order = min(plms_order, 4)  # Cap at 4th order

    # Define the adaptive cfg function with smoother transitions
    def get_effective_cfg(sigma, orig_cfg, alpha, beta, gamma, last_multiplier):
        t = sigma / sigma_max  # Normalized time parameter
        target_multiplier = 1.0 + alpha * (1 - t) ** beta - gamma * (1 - t) ** (2 * beta)

        # Limit rate of change
        max_change_rate = 0.1
        delta = target_multiplier - last_multiplier
        new_multiplier = last_multiplier + torch.clamp(torch.tensor(delta), -max_change_rate, max_change_rate).item()

        # Safety clamps to prevent extreme values
        new_multiplier = max(0.1, min(new_multiplier, 2.0))

        return orig_cfg * new_multiplier, new_multiplier

    # Define the balanced Langevin noise function
    def balanced_langevin_noise_fn(x, sigma, sigma_next, strength, aspect_ratio):
        # Scale noise based on the step size
        step_size = sigma - sigma_next
        relative_step = step_size / sigma

        # More noise at high sigmas, less at low sigmas
        noise_scale = strength * relative_step * (sigma / sigma_max)

        # Safety clamp
        noise_scale = max(0.0, min(noise_scale, 0.5))

        # Generate base noise
        noise = torch.randn_like(x) * noise_scale

        # Balance noise to preserve aspect ratio
        # Create a scaling factor tensor that's properly shaped for broadcasting
        # Channels dimension stays the same, height gets slightly boosted
        height_scale = torch.sqrt(torch.tensor(aspect_ratio))
        width_scale = 1.0 / height_scale

        # Apply to spatial dimensions only (not batch or channels)
        scaling = torch.tensor([1.0, 1.0, height_scale, width_scale]).reshape(1, -1, 1, 1).to(x.device)
        balanced_noise = noise * scaling

        return balanced_noise

    for i in trange(len(sigmas) - 1, disable=disable):
        # Apply s_churn noise if requested
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        # Calculate adaptive CFG for this step with smoother transitions
        current_cfg, last_cfg_multiplier = get_effective_cfg(
            sigma_hat, original_cfg, cfg_alpha, cfg_beta, cfg_gamma, last_cfg_multiplier
        )
        model.p.cfg_scale = current_cfg

        # Perform model prediction
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Calculate the derivative
        d = to_d(x, sigma_hat, denoised)
        # Store in history
        ds.append(d)
        if len(ds) > plms_order:
            ds.pop(0)

        # Call the callback
        if callback is not None:
            callback(
                {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised, "cfg": current_cfg}
            )

        # Step using PLMS method
        dt = sigmas[i + 1] - sigma_hat

        if len(ds) == 1:  # First step: Euler
            dx = ds[0]
        elif len(ds) == 2:  # Second-order PLMS
            dx = (3 * ds[1] - ds[0]) / 2
        elif len(ds) == 3:  # Third-order PLMS
            dx = (23 * ds[2] - 16 * ds[1] + 5 * ds[0]) / 12
        else:  # Fourth-order PLMS
            dx = (55 * ds[3] - 59 * ds[2] + 37 * ds[1] - 9 * ds[0]) / 24

        # Apply PLMS step
        x = x + dx * dt

        # Apply Langevin noise if not the final step
        if sigmas[i + 1] > 0:
            langevin_noise = balanced_langevin_noise_fn(x, sigma_hat, sigmas[i + 1], langevin_strength, aspect_ratio)
            x = x + langevin_noise

    # Reset the CFG to original value
    model.p.cfg_scale = original_cfg
    return x


# Integration functions for Automatic1111/Forge
@torch.no_grad()
def sample_langevin_plms_dynamic_cfg(
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
    langevin_strength=0.1,
    cfg_alpha=0.3,
    cfg_beta=2.0,
    cfg_gamma=0.1,
):
    """Convenience function for Langevin PLMS sampler with standard parameters."""
    return sample_langevin_plms(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        s_churn,
        s_tmin,
        s_tmax,
        s_noise,
        noise_sampler,
        langevin_strength,
        cfg_alpha,
        cfg_beta,
        cfg_gamma,
    )
