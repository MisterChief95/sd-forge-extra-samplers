import torch
from tqdm import trange

import lib_es.const as consts
from lib_es.utils import default_noise_sampler, sampler_metadata


# From ComfyUI
@sampler_metadata(
    "Extended Reverse-Time SDE",
    {"uses_ensd": True, "scheduler": "sgm_uniform"},
    ["sample_er_sde, extended_reverse_sde"],
)
@torch.no_grad()
def sample_er_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    noise_scaler=None,
):
    """
    Extended Reverse-Time SDE solver (VE ER-SDE-Solver-3). Arxiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    max_stage: int = getattr(model.p, consts.ER_MAX_STAGE, 3)

    def default_noise_scaler(sigma):
        return sigma * ((sigma**0.3).exp() + 10.0)

    noise_scaler = default_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        elif stage_used == 1:
            r = noise_scaler(sigmas[i + 1]) / noise_scaler(sigmas[i])
            x = r * x + (1 - r) * denoised
        else:
            r = noise_scaler(sigmas[i + 1]) / noise_scaler(sigmas[i])
            x = r * x + (1 - r) * denoised

            dt = sigmas[i + 1] - sigmas[i]
            sigma_step_size = -dt / num_integration_points
            sigma_pos = sigmas[i + 1] + point_indice * sigma_step_size
            scaled_pos = noise_scaler(sigma_pos)

            # Stage 2
            s = torch.sum(1 / scaled_pos) * sigma_step_size
            denoised_d = (denoised - old_denoised) / (sigmas[i] - sigmas[i - 1])
            x = x + (dt + s * noise_scaler(sigmas[i + 1])) * denoised_d

            if stage_used >= 3:
                # Stage 3
                s_u = torch.sum((sigma_pos - sigmas[i]) / scaled_pos) * sigma_step_size
                denoised_u = (denoised_d - old_denoised_d) / ((sigmas[i] - sigmas[i - 2]) / 2)
                x = x + ((dt**2) / 2 + s_u * noise_scaler(sigmas[i + 1])) * denoised_u
            old_denoised_d = denoised_d

        if s_noise != 0 and sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (
                sigmas[i + 1] ** 2 - sigmas[i] ** 2 * r**2
            ).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x
