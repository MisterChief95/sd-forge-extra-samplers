import math
import torch
from tqdm.auto import trange
from k_diffusion.sampling import to_d, get_ancestral_step
from backend.modules.k_diffusion_extra import default_noise_sampler
from lib_es.param_samplers import AdaptiveProgressiveSampling


class AdaptiveProgressSampler(AdaptiveProgressiveSampling):
    @torch.no_grad()
    def adaptive_progressive_sampler(
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
        eta=1,  # Ancestral noise parameter
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

        eta = super().get_eta()
        detail_strength = super().get_detail_strength()

        # Store previous steps' information
        prev_d = None
        prev_denoised = None

        euler_end, dpm_end = get_phase_bounds(steps)

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
            step_eta = eta if progress < 0.5 else eta * (1.0 - min(1.0, (progress - 0.5) * 2.0))
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


def get_phase_bounds(steps: int) -> tuple[float, float]:
    """
    Calculate phase boundaries for the adaptive progressive sampler.

    Args:
        steps: Total number of steps

    Returns:
        Tuple of phase boundaries (Euler end, DPM++ end)
    """
    if steps < 10:
        return 0.20, 0.70  # 20% Euler, 50% DPM++, 30% Detail
    if steps < 20:
        return 0.20, 0.70  # 20% Euler, 50% DPM++, 30% Detail
    if steps < 40:
        return 0.25, 0.70  # 25% Euler, 45% DPM++, 30% Detail
    return 0.30, 0.75  # 30% Euler, 45% DPM++, 25% Detail


@torch.no_grad()
def sample_adaptive_progress_sampler(
    model,
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
    return AdaptiveProgressSampler().adaptive_progressive_sampler(
        model, x, sigmas, extra_args, callback, disable, s_churn, s_tmin, s_tmax, eta, s_noise, noise_sampler
    )
