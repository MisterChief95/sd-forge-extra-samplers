import torch
from tqdm.auto import trange

from modules_forge.packages.k_diffusion.sampling import (
    to_d,
    sigma_to_half_log_snr,
    offset_first_sigma_for_snr,
)
from lib_es.extra_samplers.utils.sa_solver import (
    compute_stochastic_adams_b_coeffs,
    get_tau_interval_func,
)
from lib_es.utils import sampler_metadata, default_noise_sampler


@torch.no_grad()
@sampler_metadata("SA-Solver", {"uses_ensd": True, "scheduler": "karras"})
def sample_sa_solver(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    tau_func=None,
    s_noise=1.0,
    noise_sampler=None,
    predictor_order=3,
    corrector_order=4,
    use_pece=False,
    simple_order_2=False,
):
    """Stochastic Adams Solver with predictor-corrector method (NeurIPS 2023)."""
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = sigma_to_half_log_snr(sigmas, model_sampling=model_sampling)

    if tau_func is None:
        # Use default interval for stochastic sampling
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = get_tau_interval_func(start_sigma, end_sigma, eta=1.0)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x  # x: current state, x_pred: predicted next state

    h = x.new_tensor(0.0)
    tau_t = 0.0
    noise = 0.0
    pred_list = []

    # Lower order near the end to improve stability
    lower_order_to_end = sigmas[-1].item() == 0

    for i in trange(len(sigmas) - 1, disable=disable):
        # Evaluation
        denoised = model(x_pred, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x_pred, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        # Corrector
        if corrector_order_used == 0:
            # Update by the predicted state
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1 : i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)  # (B, K, ...)
            corr_res = torch.einsum("bk...,k->b...", pred_mat, b_coeffs)  # (B, ...)
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t**2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                # The noise from the previous predictor step
                x = x + noise

            if use_pece:
                # Evaluate the corrected state
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                pred_list[-1] = denoised

        # Predictor
        if sigmas[i + 1] == 0:
            # Denoising step
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = lambdas[i - predictor_order_used + 1 : i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)  # (B, K, ...)
            pred_res = torch.einsum("bk...,k->b...", pred_mat, b_coeffs)  # (B, ...)
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t**2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = (
                    noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * tau_t**2 * h).expm1().neg().sqrt()
                    * s_noise
                )
                x_pred = x_pred + noise
    return x_pred


@torch.no_grad()
@sampler_metadata("SA-Solver PECE", {"uses_ensd": True, "scheduler": "karras"})
def sample_sa_solver_pece(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    tau_func=None,
    s_noise=1.0,
    noise_sampler=None,
    predictor_order=3,
    corrector_order=4,
    simple_order_2=False,
):
    """Stochastic Adams Solver with PECE (Predict-Evaluate-Correct-Evaluate) mode (NeurIPS 2023)."""
    return sample_sa_solver(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        tau_func=tau_func,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        use_pece=True,
        simple_order_2=simple_order_2,
    )


# Original sampler by https://github.com/eddyhhlure1Eddy/ode-ComfyUI-WanVideoWrapper
# Adapted into ComfyUI by https://github.com/kabachuha/ComfyUI-SA-ODE-Stable-Sampler
# Apache 2.0 License
@torch.no_grad()
@sampler_metadata("SA-Solver Stable", {"uses_ensd": True, "scheduler": "karras"})
def sample_sa_solver_stable(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    solver_order=3,
    use_adaptive_order=True,
    use_velocity_smoothing=True,
    convergence_threshold=0.15,
    smoothing_factor=0.8,
):
    """Deterministic SA-ODE stable-convergence variant."""
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    velocity_buffer = []
    smoothed_velocity = None
    num_inference_steps = len(sigmas)

    def get_adaptive_order(sigma):
        if not use_adaptive_order:
            return solver_order

        if num_inference_steps <= 8:
            return min(2, solver_order)

        if sigma > 0.7:
            return min(2, solver_order)
        if sigma > convergence_threshold:
            return solver_order
        return max(1, solver_order - 1)

    def compute_multistep_velocity(order):
        if not velocity_buffer:
            raise RuntimeError("velocity_buffer is empty")

        order = min(order, len(velocity_buffer))
        if order >= 3 and len(velocity_buffer) >= 3:
            return (23 / 12) * velocity_buffer[-1] - (16 / 12) * velocity_buffer[-2] + (5 / 12) * velocity_buffer[-3]
        if order >= 2 and len(velocity_buffer) >= 2:
            return 1.5 * velocity_buffer[-1] - 0.5 * velocity_buffer[-2]
        if len(velocity_buffer) >= 1:
            return velocity_buffer[-1]
        raise RuntimeError("No velocity data available")

    def apply_velocity_smoothing(velocity, sigma):
        nonlocal smoothed_velocity
        if not use_velocity_smoothing:
            return velocity

        if num_inference_steps <= 8:
            return velocity

        if sigma < convergence_threshold:
            if smoothed_velocity is None:
                smoothed_velocity = velocity
            else:
                alpha = smoothing_factor
                smoothed_velocity = alpha * smoothed_velocity + (1 - alpha) * velocity
            return smoothed_velocity

        smoothed_velocity = velocity
        return velocity

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)
        d = to_d(x, sigma, denoised)

        velocity_buffer.append(d)
        while len(velocity_buffer) > solver_order + 1:
            velocity_buffer.pop(0)

        current_order = get_adaptive_order(sigma.item())
        velocity = compute_multistep_velocity(current_order) if len(velocity_buffer) >= 2 else d
        velocity = apply_velocity_smoothing(velocity, sigma.item())

        dt = sigma_next - sigma
        if num_inference_steps > 8 and sigma.item() < convergence_threshold:
            damping = 0.5 + 0.5 * (sigma.item() / convergence_threshold)
            dt = dt * damping

        if sigma_next == 0:
            x = denoised
        else:
            x = x + velocity * dt

        if num_inference_steps > 8 and sigma.item() < 0.05 and len(velocity_buffer) >= 3:
            avg_velocity = sum(velocity_buffer[-3:]) / 3
            stabilized = x + avg_velocity * dt
            blend_factor = sigma.item() / 0.05
            x = blend_factor * x + (1 - blend_factor) * stabilized

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

    return x
