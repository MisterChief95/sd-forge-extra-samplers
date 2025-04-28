from collections.abc import Callable
from typing import Any, Optional
import torch
from tqdm import trange

from k_diffusion.sampling import to_d
from modules import errors

import lib_es.const as consts
from lib_es.utils import sampler_metadata


def compute_optimal_gamma(steps: int, adaptive: bool = True) -> float:
    """
    Compute the optimal gamma parameter for gradient estimation based on step count.

    Args:
        steps: Number of sampling steps
        adaptive: Whether to use adaptive gamma based on step count

    Returns:
        Optimal gamma value
    """
    if not adaptive:
        return consts.GE_DEFAULT_GAMMA

    # Define min and max values
    min_steps, max_steps = 10, 100
    min_gamma, max_gamma = 1.5, 2.6

    # Handle edge cases
    if steps <= min_steps:
        return min_gamma
    elif steps >= max_steps:
        return max_gamma

    # Apply logarithmic scaling
    # log(steps/min_steps) / log(max_steps/min_steps) gives a value from 0 to 1
    # that increases logarithmically with steps
    log_factor = torch.log(torch.tensor(steps / min_steps)) / torch.log(torch.tensor(max_steps / min_steps))

    # Convert the logarithmic factor to gamma value
    gamma = min_gamma + log_factor * (max_gamma - min_gamma)

    return float(gamma)


def validate_schedule(sigmas: torch.Tensor, eta: float = 0.1, nu: float = 2.0) -> bool:
    """
    Validate whether a noise schedule satisfies the admissibility criteria from the paper.

    Args:
        sigmas: Tensor of noise levels in descending order
        eta: Error parameter
        nu: Accuracy parameter for distance estimates

    Returns:
        True if schedule is admissible, False otherwise
    """
    n = len(sigmas) - 1
    is_admissible = True
    issues = []

    # Check if sigmas are strictly decreasing
    if not torch.all(sigmas[:-1] > sigmas[1:]):
        is_admissible = False
        issues.append("Sigmas must be strictly decreasing")

    # Calculate the maximum allowable beta
    c = 1 - nu ** (-1 / n)
    beta_max = c / (eta + c)

    # Check that step sizes respect the admissibility criteria
    for i in range(n - 1):
        ratio = sigmas[i + 1] / sigmas[i]
        beta = 1 - ratio
        if beta > beta_max:
            is_admissible = False
            issues.append(f"Step {i} has beta {beta:.4f} > beta_max {beta_max:.4f}")

    if not is_admissible:
        errors.display(ValueError(f"Noise schedule is not admissible: {', '.join(issues)}"))
        errors.print_error_explanation(f"Noise schedule validation failed.\n\tIssues: {',\n\t\t'.join(issues)}")

    return is_admissible


@torch.no_grad()
@sampler_metadata("Gradient Estimation", {"scheduler": "sgm_uniform"})
def sample_gradient_estimation(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    validate_sigmas: bool = False,
    eta: float = 0.1,
    nu: float = 2.0,
) -> torch.Tensor:
    """
    Gradient-estimation sampler as described in "Interpreting and Improving Diffusion Models from an Optimization Perspective".

    This sampler implements a second-order method that improves upon DDIM by using a combination of current and previous
    gradients to reduce gradient estimation error. It is based on the insight that denoising is approximately equivalent to
    projection onto the data manifold, and diffusion sampling is gradient descent on the squared Euclidean distance function.

    Args:
        model: The diffusion model
        x: Input tensor
        sigmas: Noise schedule (should be in descending order)
        extra_args: Extra arguments to pass to the model
        callback: Callback function
        disable: Whether to disable the progress bar
        validate_sigmas: Whether to validate the noise schedule
        eta: Error parameter for schedule validation (default 0.1)
        nu: Accuracy parameter for schedule validation (default 2.0)

    Returns:
        Denoised tensor

    References:
        Paper: https://openreview.net/pdf?id=o2ND9v0CeK
    """
    # Parameter validation and initialization
    if sigmas.shape[0] < 2:
        raise ValueError("Need at least 2 timesteps for gradient estimation")

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None
    steps = len(sigmas) - 1

    # Schedule validation
    if validate_sigmas:
        validate_schedule(sigmas, eta, nu)

    # Get gamma from model properties or compute optimal value
    use_adaptive_steps: bool = getattr(model.p, consts.GE_USE_ADAPTIVE_STEPS, True)
    if use_adaptive_steps:
        # Compute optimal gamma based on the number of steps
        # and add the offset if specified
        ge_gamma = compute_optimal_gamma(steps, use_adaptive_steps) + getattr(
            model.p, consts.GE_GAMMA_OFFSET, consts.GE_DEFAULT_GAMMA_OFFSET
        )
    else:
        ge_gamma = getattr(model.p, consts.GE_GAMMA, consts.GE_DEFAULT_GAMMA)

    # Initialize timestep-adaptive gamma values if needed
    timestep_adaptive_gamma = getattr(model.p, consts.GE_USE_TIMESTEP_ADAPTIVE_GAMMA, False)

    if timestep_adaptive_gamma:
        # Higher gamma at the beginning, lower toward the end
        # This is a heuristic based on the observation that early steps benefit more
        # from aggressive gradient correction
        gammas = torch.linspace(ge_gamma * 1.2, ge_gamma * 0.8, steps)

    # Main sampling loop
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        dt = sigmas[i + 1] - sigmas[i]

        if i == 0:
            # Euler method for first step
            x = x + d * dt
        else:
            # Gradient estimation
            current_gamma = gammas[i].item() if timestep_adaptive_gamma else ge_gamma

            d_bar = current_gamma * d + (1 - current_gamma) * old_d
            x = x + d_bar * dt

        old_d = d

    return x
