import torch
from tqdm.auto import trange
from functools import partial

from modules_forge.packages.k_diffusion.sampling import (
    sigma_to_half_log_snr,
    half_log_snr_to_sigma,
    offset_first_sigma_for_snr,
)
from lib_es.utils import (
    sampler_metadata,
    default_noise_sampler,
    ei_h_phi_1,
    ei_h_phi_2,
)
from lib_es.const import SEEDS_3_R1, SEEDS_3_R2, SEEDS_3_DEFAULT_R1, SEEDS_3_DEFAULT_R2


@torch.no_grad()
@sampler_metadata("SEEDS-3", {"uses_ensd": True, "scheduler": "karras", "sampler_order": 3})
def sample_seeds_3(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r_1=1.0 / 3,
    r_2=2.0 / 3,
):
    """SEEDS-3 - Stochastic Explicit Exponential Derivative-free Solvers (VP Data Prediction) stage 3.
    arXiv: https://arxiv.org/abs/2305.14267 (NeurIPS 2023)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    inject_noise = eta > 0 and s_noise > 0

    model_sampling = model.inner_model.predictor
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

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
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r_1)
        lambda_s_2 = torch.lerp(lambda_s, lambda_t, r_2)
        sigma_s_1, sigma_s_2 = sigma_fn(lambda_s_1), sigma_fn(lambda_s_2)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_s_2 = sigma_s_2 * lambda_s_2.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1
        x_2 = sigma_s_1 / sigmas[i] * (-r_1 * h * eta).exp() * x - alpha_s_1 * ei_h_phi_1(-r_1 * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r_1 * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        # Step 2
        a3_2 = r_2 / r_1 * ei_h_phi_2(-r_2 * h_eta)
        a3_1 = ei_h_phi_1(-r_2 * h_eta) - a3_2
        x_3 = sigma_s_2 / sigmas[i] * (-r_2 * h * eta).exp() * x - alpha_s_2 * (a3_1 * denoised + a3_2 * denoised_2)
        if inject_noise:
            segment_factor = (r_1 - r_2) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigma_s_2)
            x_3 = x_3 + sde_noise * sigma_s_2 * s_noise
        denoised_3 = model(x_3, sigma_s_2 * s_in, **extra_args)

        # Step 3
        b3 = ei_h_phi_2(-h_eta) / r_2
        b1 = ei_h_phi_1(-h_eta) - b3
        x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * (b1 * denoised + b3 * denoised_3)
        if inject_noise:
            segment_factor = (r_2 - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_2, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x
