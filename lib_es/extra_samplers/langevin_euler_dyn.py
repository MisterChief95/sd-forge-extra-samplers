import torch
from tqdm.auto import trange
from k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d
from modules.shared import opts


@torch.no_grad()
def sample_langevin(
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
):
    """
    Langevin dynamics sampler - the adaptive CFG is now handled by the CFG function.
    This is your original implementation but with the adaptive CFG logic removed.
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # Store original shape for aspect ratio calculations
    height, width = x.shape[2:4]
    aspect_ratio = width / height
    sigma_max = sigmas[0]

    for i in trange(len(sigmas) - 1, disable=disable):
        # Apply s_churn noise if requested
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        # Perform model prediction - CFG is now handled by our function
        denoised = model(x, sigma_hat * s_in, **extra_args)

        # Call the callback
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Calculate the derivative
        d = to_d(x, sigma_hat, denoised)

        # Langevin step: Deterministic part + Noise part
        dt = sigmas[i + 1] - sigma_hat

        # Deterministic Euler step
        x = x + d * dt

        # Apply Langevin noise if not the final step
        if sigmas[i + 1] > 0:
            # Simpler Langevin noise logic with less aggressive scaling
            # Use a constant base noise level with a gentle decay
            base_noise_level = langevin_strength  # Base level from parameter

            # Gentle decay curve - more consistent noise across steps
            # Sqrt provides a more gradual decrease than linear scaling
            decay_factor = torch.sqrt(sigmas[i + 1] / sigma_max)
            noise_scale = base_noise_level * (0.1 + 0.9 * decay_factor)

            # Higher safety clamp to allow more noise influence
            noise_scale = max(langevin_strength * 0.05, min(noise_scale, 0.8))

            # Generate balanced noise
            noise = torch.randn_like(x) * noise_scale
            height_scale = torch.sqrt(torch.tensor(aspect_ratio))
            width_scale = 1.0 / height_scale
            scaling = torch.tensor([1.0, 1.0, height_scale, width_scale]).reshape(1, -1, 1, 1).to(x.device)
            balanced_noise = noise * scaling

            x = x + balanced_noise

    return x


@torch.no_grad()
def sample_langevin_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    langevin_strength=0.1,
):
    """
    Ancestral sampling with Langevin dynamics.
    The adaptive CFG is now handled by the CFG function set externally.
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # Store original shape for aspect ratio calculations
    height, width = x.shape[2:4]
    aspect_ratio = width / height
    sigma_max = sigmas[0]

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
        height_scale = torch.sqrt(torch.tensor(aspect_ratio))
        width_scale = 1.0 / height_scale

        # Apply to spatial dimensions only
        scaling = torch.tensor([1.0, 1.0, height_scale, width_scale]).reshape(1, -1, 1, 1).to(x.device)
        balanced_noise = noise * scaling

        return balanced_noise

    for i in trange(len(sigmas) - 1, disable=disable):
        # Perform model prediction - CFG is now handled by our function
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        # Ancestral sampling calculation
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        # Call the callback
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        # Calculate the derivative
        d = to_d(x, sigmas[i], denoised)

        # Deterministic Euler step
        dt = sigma_down - sigmas[i]
        x = x + d * dt

        # Apply Langevin noise if not the final step (separate from ancestral noise)
        if sigmas[i + 1] > 0:
            # Add Langevin noise
            langevin_noise = balanced_langevin_noise_fn(x, sigmas[i], sigma_down, langevin_strength, aspect_ratio)
            x = x + langevin_noise

            # Add ancestral noise - also apply aspect ratio balancing
            ancestral_noise = noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            height_scale = torch.sqrt(torch.tensor(aspect_ratio))
            width_scale = 1.0 / height_scale
            scaling = torch.tensor([1.0, 1.0, height_scale, width_scale]).reshape(1, -1, 1, 1).to(x.device)
            balanced_ancestral_noise = ancestral_noise * scaling
            x = x + balanced_ancestral_noise

    return x


# Integration functions for Automatic1111/Forge
@torch.no_grad()
def sample_langevin_euler_dynamic_cfg(
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
    """Convenience function for Langevin sampler with adaptive CFG."""

    # TODO: Handle XYZ
    return sample_langevin(
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
        langevin_strength=opts.data.get("exs_langevin_strength", 0.1),
    )


@torch.no_grad()
def sample_langevin_ancestral_euler_dynamic_cfg(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """Convenience function for ancestral Langevin sampler with adaptive CFG."""

    # TODO: Handle XYZ
    return sample_langevin_ancestral(
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        eta=1.0,
        s_noise=1.0,
        noise_sampler=None,
        langevin_strength=opts.data.get("exs_langevin_strength", 0.1),
    )
