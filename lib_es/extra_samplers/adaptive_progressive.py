import math
import torch
from tqdm.auto import trange
from k_diffusion.sampling import to_d, get_ancestral_step
from backend.modules.k_diffusion_extra import default_noise_sampler

import lib_es.const as consts


@torch.no_grad()
def sample_adaptive_progressive(
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
    Adaptive progressive sampler that automatically adjusts to different step counts.
    Combines Euler ancestral, DPM++ 2M, and detail enhancement with phase-based transitions.

    This sampler is optimized for both high and very low step counts (4+),
    dynamically adjusting phase durations based on total step count.

    Args:
        model: The denoising model
        x: Input noise tensor
        sigmas: Noise schedule
        extra_args: Additional arguments for the model
        callback: Optional callback function
        disable: Whether to disable the progress bar
        s_churn: Amount of stochasticity
        s_tmin: Minimum sigma for stochasticity
        s_tmax: Maximum sigma for stochasticity
        eta: Ancestral noise parameter
        s_noise: Noise scale
        noise_sampler: Custom noise sampler function
        detail_strength: Strength of detail enhancement phase

    Returns:
        Denoised tensor
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    steps = len(sigmas) - 1

    euler_a_end = getattr(model.p, consts.AP_EULER_A_END, 0.35)
    dpm_2m_end = getattr(model.p, consts.AP_DPM_2M_END, 0.75)
    ancestral_eta = getattr(model.p, consts.AP_ANCESTRAL_ETA, 0.4)
    detail_strength = getattr(model.p, consts.AP_DETAIL_STRENGTH, 1.5)

    # Store previous steps' information
    prev_d = None
    prev_denoised = None

    euler_end, dpm_end = calc_phase_bounds(steps, euler_a_end, dpm_2m_end)

    for i in trange(steps, disable=disable):
        progress = i / steps

        # Calculate weights based on phase
        if progress < euler_end:
            # Euler ancestral phase
            w_euler = 1.0
            w_multi = 0.0
            w_detail = 0.0
        elif progress < dpm_end:
            # DPM++ phase - smooth transition from Euler
            phase_progress = (progress - euler_end) / (dpm_end - euler_end)
            w_euler = max(0.0, 1.0 - phase_progress * 2.5)  # Faster transition out of Euler
            w_multi = 1.0 - w_euler
            w_detail = 0.0
        else:
            # Detail refinement phase - gradual transition
            phase_progress = (progress - dpm_end) / (1.0 - dpm_end)
            w_euler = 0.0
            w_multi = max(0.0, 1.0 - phase_progress * 1.5)  # Gradual reduction in DPM++
            w_detail = 1.0 - w_multi

        # Apply adaptive stochasticity (only in early steps)
        if s_churn > 0 and s_tmin <= sigmas[i] <= s_tmax and progress < 0.4:
            # Scale down stochasticity as we progress
            gamma = min(s_churn / steps, 2**0.5 - 1) * (1.0 - progress / 0.4)
            sigma_hat = sigmas[i] * (gamma + 1)
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2).sqrt()
        else:
            sigma_hat = sigmas[i]

        # Get denoised prediction
        denoised = model(x, sigma_hat * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Calculate sigma for step
        # Reduce eta as we progress to lower noise in later steps
        step_eta = ancestral_eta if progress < 0.5 else ancestral_eta * (1.0 - min(1.0, (progress - 0.5) * 2.0))
        sigma_down, sigma_up = get_ancestral_step(sigma_hat, sigmas[i + 1], eta=step_eta)

        # Calculate current score
        d = to_d(x, sigma_hat, denoised)
        dt = sigma_down - sigma_hat

        # Special case for final step
        if sigmas[i + 1] == 0:
            x = denoised
            break

        # Calculate step direction based on phase
        if prev_d is None:
            # First step is pure Euler ancestral
            direction = d
        else:
            # Initialize direction
            direction = torch.zeros_like(d)

            # Add Euler component if needed
            if w_euler > 0:
                direction += w_euler * d

            # Add DPM++ component if needed
            if w_multi > 0:
                # Adjust coefficients based on noise level
                if sigma_hat > 2.0:
                    # Higher noise: favor current direction
                    c1, c2 = 0.7, 0.3
                else:
                    # Lower noise: more balanced
                    c1, c2 = 0.6, 0.4

                multi_direction = c1 * d + c2 * prev_d
                direction += w_multi * multi_direction

            # Add detail enhancement if needed
            if w_detail > 0 and prev_denoised is not None:
                # Only apply significant enhancement at lower noise levels
                if sigma_hat < 1.0:
                    # Calculate detail vector (high frequency components)
                    detail_vector = denoised - prev_denoised

                    # Scale based on noise level - stronger at very low noise
                    detail_scale = detail_strength * min(1.0, 0.2 / (sigma_hat + 0.2))

                    # Apply detail enhancement with adaptive scaling
                    detail_direction = d + detail_vector * detail_scale / dt
                    direction += w_detail * detail_direction
                else:
                    # At higher noise levels, use standard direction
                    direction += w_detail * d

        # Ensure numerical stability
        direction = torch.clamp(direction, -1e2, 1e2)

        # Apply the step
        x = x + direction * dt

        # Apply ancestral noise with progressive reduction
        if sigma_up > 0:
            # Only add significant noise in earlier steps
            noise_scale = s_noise
            if progress > 0.3:
                # Exponential reduction in noise after Euler phase
                noise_scale *= math.exp(-4.0 * (progress - 0.3))

            # Add the scaled noise
            x = x + noise_sampler(sigma_hat, sigmas[i + 1]) * sigma_up * noise_scale

        # Store values for next step
        prev_d = d
        prev_denoised = denoised

    return x


def calc_phase_bounds(steps: int, custom_euler_end: float = 0.25, custom_dpm_end: float = 0.7) -> tuple[float, float]:
    """
    Calculate phase boundaries for the adaptive progressive sampler.

    Args:
        steps: Total number of steps
        custom_euler_end: End point for Euler phase (0.0-1.0)
        custom_dpm_end: End point for DPM++ phase (0.0-1.0)

    Returns:
        Tuple of phase boundaries (Euler end, DPM++ end)
    """
    # Ensure values are within valid range
    euler_end = max(0.0, min(1.0, custom_euler_end))
    dpm_end = max(0.0, min(1.0, custom_dpm_end))

    # Ensure euler_end < dpm_end
    if euler_end >= dpm_end:
        euler_end = max(0.0, dpm_end - 0.2)  # Ensure at least 20% for DPM++ phase

    # Adaptive phase boundaries based on step count
    if steps < 10:
        # For very low step counts, shorten Euler phase and extend detail phase
        euler_end = min(euler_end, 0.15 + (steps - 4) * 0.01)
        dpm_end = min(dpm_end, 0.5 + (steps - 4) * 0.02)
    elif steps < 20:
        # For low step counts, slightly adjust phases
        euler_end = min(euler_end, 0.2)
        dpm_end = min(dpm_end, 0.65)
    elif steps > 50:
        # For high step counts, extend the Euler phase slightly
        euler_end = min(0.3, euler_end + (steps - 50) * 0.0005)
        # And allow for a longer DPM++ phase
        dpm_end = min(0.8, dpm_end + (steps - 50) * 0.0005)

    # Ensure minimum phase lengths
    if dpm_end - euler_end < 0.1:
        dpm_end = min(1.0, euler_end + 0.1)

    return euler_end, dpm_end
