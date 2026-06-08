import torch
from tqdm.auto import trange
from functools import partial

from modules_forge.packages.k_diffusion.sampling import (
    sigma_to_half_log_snr,
    half_log_snr_to_sigma,
    offset_first_sigma_for_snr,
)
from lib_es.utils import (
    default_noise_sampler,
    sampler_metadata,
    ei_h_phi_1,
    ei_h_phi_2,
)


@torch.no_grad()
@sampler_metadata("SEEDS-2", {"uses_ensd": True, "scheduler": "karras", "sampler_order": 2})
def sample_seeds_2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r=0.5,
    solver_type="phi_1",
):
    """SEEDS-2 - Stochastic Explicit Exponential Derivative-free Solvers (VP Data Prediction) stage 2.
    arXiv: https://arxiv.org/abs/2305.14267 (NeurIPS 2023)
    """
    if solver_type not in {"phi_1", "phi_2"}:
        raise ValueError("solver_type must be 'phi_1' or 'phi_2'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    inject_noise = eta > 0 and s_noise > 0

    model_sampling = model.inner_model.predictor
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    fac = 1 / (2 * r)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r)
        sigma_s_1 = sigma_fn(lambda_s_1)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1
        x_2 = sigma_s_1 / sigmas[i] * (-r * h * eta).exp() * x - alpha_s_1 * ei_h_phi_1(-r * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        # Step 2
        if solver_type == "phi_1":
            denoised_d = torch.lerp(denoised, denoised_2, fac)
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * ei_h_phi_1(-h_eta) * denoised_d
        elif solver_type == "phi_2":
            b2 = ei_h_phi_2(-h_eta) / r
            b1 = ei_h_phi_1(-h_eta) - b2
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * (b1 * denoised + b2 * denoised_2)

        if inject_noise:
            segment_factor = (r - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x
