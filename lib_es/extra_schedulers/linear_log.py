import torch

from lib_es.utils import scheduler_metadata


@scheduler_metadata(name="linear_log", alias="Linear Log", need_inner_model=True)
def linear_log(
    n: int,
    sigma_min: float,
    sigma_max: float,
    inner_model,
    device: torch.device,
    eta: float = 0.1,
    nu: float = 2.0,
    sgm: bool = False,
    floor=False,
    final_step_full: bool = True,
) -> torch.Tensor:
    """
    Creates a log-linear (geometric) noise schedule as recommended in the paper.

    Args:
        n: Number of sampling steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        inner_model: Inner model
        device: Device to place the tensor on
        eta: Error parameter (default 0.1, as estimated in the paper for CIFAR-10)
        nu: Accuracy parameter for distance estimates (default 2.0)
        sgm: Whether to include the final sigma=0-step
        floor: Whether to floor sigma values at sigma_min
        final_step_full: Whether to take a full step (β=1) for the final iteration

    Returns:
        A tensor of sigma values in descending order with a geometric progression.
    """

    # TODO: Add adjustable eta/nu parameters for more flexibility

    # Calculate the maximum allowable beta based on the admissibility criteria
    # β*,N = c/(η+c) where c = 1 - ν^(-1/N)
    c = 1 - nu ** (-1 / n)
    beta_max = c / (eta + c)

    # Calculate the ratio that would give us exactly sigma_min from sigma_max in n steps
    exact_ratio = (sigma_min / sigma_max) ** (1 / (n - 1))

    # Use the smaller of the two to ensure admissibility
    ratio = max(1 - beta_max, exact_ratio)

    # Generate the geometric sequence
    sigs = [sigma_max]
    for i in range(1, n):
        next_sigma = sigs[-1] * ratio

        # For the final step, optionally set beta=1 (as recommended in the paper)
        if final_step_full and i == n - 1:
            next_sigma = sigma_min

        sigs.append(next_sigma)

    if not sgm:
        # Add final value of 0.0
        sigs.append(0.0)

    # Convert to tensor
    return torch.tensor(sigs)
