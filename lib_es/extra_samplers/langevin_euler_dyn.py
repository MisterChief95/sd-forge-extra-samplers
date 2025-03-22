import torch
from tqdm.auto import trange
from k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d

from lib_es.param_samplers import DynamicCFGSampler, LangevinSampler


class LangevinEulerDynamicCFGSampler(DynamicCFGSampler, LangevinSampler):
    @torch.no_grad()
    def sample_langevin(
        self,
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
        """
        Langevin dynamics sampler with adaptive CFG.

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
        """
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        s_in = x.new_ones([x.shape[0]])

        cfg_alpha, cfg_beta, cfg_gamma = super().get_cfg_values()
        langevin_strength = super().get_langevin_strength()

        model.p.extra_generation_params.update(
            {
                "cfg_alpha": cfg_alpha,
                "cfg_beta": cfg_beta,
                "cfg_gamma": cfg_gamma,
                "langevin_strength": langevin_strength,
            }
        )

        # Store original cfg scale
        original_cfg = getattr(model.p, "cfg_scale", 7.0)
        sigma_max = sigmas[0]

        # Store original shape for aspect ratio calculations
        height, width = x.shape[2:4]
        aspect_ratio = width / height

        # Initialize for CFG smoothing
        last_cfg_multiplier = 1.0

        # Define the adaptive cfg function with smoother transitions
        def get_effective_cfg(sigma, orig_cfg, alpha, beta, gamma, last_multiplier):
            t = sigma / sigma_max  # Normalized time parameter
            target_multiplier = 1.0 + alpha * (1 - t) ** beta - gamma * (1 - t) ** (2 * beta)

            # Limit rate of change
            max_change_rate = 0.1
            delta = target_multiplier - last_multiplier
            new_multiplier = (
                last_multiplier + torch.clamp(torch.tensor(delta), -max_change_rate, max_change_rate).item()
            )

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

            # Call the callback
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                        "cfg": current_cfg,
                    }
                )

            # Calculate the derivative
            d = to_d(x, sigma_hat, denoised)

            # Langevin step: Deterministic part + Noise part
            dt = sigmas[i + 1] - sigma_hat

            # Deterministic Euler step
            x = x + d * dt

            # Apply Langevin noise if not the final step
            if sigmas[i + 1] > 0:
                langevin_noise = balanced_langevin_noise_fn(
                    x, sigma_hat, sigmas[i + 1], langevin_strength, aspect_ratio
                )
                x = x + langevin_noise

        # Reset the CFG to original value
        model.p.cfg_scale = original_cfg
        return x


class LangevinEulerAncestralDynamicCFGSampler(DynamicCFGSampler, LangevinSampler):
    @torch.no_grad()
    def sample_langevin_ancestral(
        self,
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
        """
        Ancestral sampling with Langevin dynamics and adaptive CFG.

        Args:
            model: The model to sample from.
            x: The initial noise.
            sigmas: The noise levels to sample at.
            extra_args: Extra arguments to the model.
            callback: A function that's called after each step.
            disable: Disable tqdm progress bar.
            eta: Ancestral sampling strength.
            s_noise: Noise scale.
            noise_sampler: A function that returns noise.
            langevin_strength: Strength of the Langevin noise term.
        """

        extra_args = {} if extra_args is None else extra_args
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        s_in = x.new_ones([x.shape[0]])

        cfg_alpha, cfg_beta, cfg_gamma = super().get_cfg_values()
        langevin_strength = super().get_langevin_strength()

        model.p.extra_generation_params.update(
            {
                "cfg_alpha": cfg_alpha,
                "cfg_beta": cfg_beta,
                "cfg_gamma": cfg_gamma,
                "langevin_strength": langevin_strength,
            }
        )

        # Store original cfg scale
        original_cfg = getattr(model.p, "cfg_scale", 7.0)
        sigma_max = sigmas[0]

        # Store original shape for aspect ratio calculations
        height, width = x.shape[2:4]
        aspect_ratio = width / height

        # Initialize for CFG smoothing
        last_cfg_multiplier = 1.0

        # Define the adaptive cfg function with smoother transitions
        def get_effective_cfg(sigma, orig_cfg, alpha, beta, gamma, last_multiplier):
            t = sigma / sigma_max  # Normalized time parameter
            target_multiplier = 1.0 + alpha * (1 - t) ** beta - gamma * (1 - t) ** (2 * beta)

            # Limit rate of change
            max_change_rate = 0.1
            delta = target_multiplier - last_multiplier
            new_multiplier = (
                last_multiplier + torch.clamp(torch.tensor(delta), -max_change_rate, max_change_rate).item()
            )

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
            # Calculate adaptive CFG for this step with smoother transitions
            current_cfg, last_cfg_multiplier = get_effective_cfg(
                sigmas[i], original_cfg, cfg_alpha, cfg_beta, cfg_gamma, last_cfg_multiplier
            )
            model.p.cfg_scale = current_cfg

            # Perform model prediction
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
                        "cfg": current_cfg,
                    }
                )

            # Calculate the derivative
            d = to_d(x, sigmas[i], denoised)

            # Deterministic Euler step
            dt = sigma_down - sigmas[i]
            x = x + d * dt

            # Apply Langevin noise if not the final step (separate from ancestral noise)
            if sigmas[i + 1] > 0:
                langevin_noise = balanced_langevin_noise_fn(x, sigmas[i], sigma_down, langevin_strength, aspect_ratio)
                x = x + langevin_noise

                # Add ancestral noise - also apply aspect ratio balancing to ancestral noise
                ancestral_noise = noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                height_scale = torch.sqrt(torch.tensor(aspect_ratio))
                width_scale = 1.0 / height_scale
                scaling = torch.tensor([1.0, 1.0, height_scale, width_scale]).reshape(1, -1, 1, 1).to(x.device)
                balanced_ancestral_noise = ancestral_noise * scaling
                x = x + balanced_ancestral_noise

        # Reset the CFG to original value
        model.p.cfg_scale = original_cfg
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
    """Convenience function for Langevin sampler with standard parameters."""
    return LangevinEulerDynamicCFGSampler().sample_langevin(
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
    """Convenience function for ancestral Langevin sampler with standard parameters."""
    return LangevinEulerAncestralDynamicCFGSampler().sample_langevin_ancestral(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        eta,
        s_noise,
        noise_sampler,
    )
