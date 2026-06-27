import torch

from lib_es.utils import scheduler_metadata
from modules.sd_schedulers import beta_scheduler
from modules.shared import opts


@scheduler_metadata(name="beta_57", alias="Beta 57", need_inner_model=True)
def beta_57(
    n: int,
    sigma_min: float,
    sigma_max: float,
    inner_model,
    device: torch.device,
) -> torch.Tensor:
    current_alpha = opts.beta_dist_alpha
    current_beta = opts.beta_dist_beta

    opts.beta_dist_alpha = 0.5
    opts.beta_dist_beta = 0.7

    tensor = beta_scheduler(n, sigma_min, sigma_max, inner_model, device)

    opts.beta_dist_alpha = current_alpha
    opts.beta_dist_beta = current_beta
    
    return tensor
