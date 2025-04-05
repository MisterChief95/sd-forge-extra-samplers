import torch
from tqdm.auto import trange
from k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d


@torch.no_grad()
def sample_heun_ancestral(
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
    Ancestral sampling with Heun's method steps.

    Args:
        model: The model to sample from.
        x: The initial noise.
        sigmas: The noise levels to sample at.
        extra_args: Extra arguments to the model.
        callback: A function that's called after each step.
        disable: Disable tqdm progress bar.
        eta: Ancestral sampling strength parameter.
        s_noise: Noise scale.
        noise_sampler: A function that returns noise.
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        # Get current and next sigma
        sigma = sigmas[i]

        # Run denoising model
        denoised = model(x, sigma * s_in, **extra_args)

        # Calculate ancestral step parameters
        sigma_down, sigma_up = get_ancestral_step(sigma, sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        # Calculate the derivative
        d = to_d(x, sigma, denoised)

        # Determine step size
        dt = sigma_down - sigma

        if sigma_down == 0:
            # For the last step, use Euler method for stability
            x = x + d * dt
        else:
            # Heun's method (predictor-corrector)
            # 1. Predictor step (Euler)
            x_2 = x + d * dt

            # 2. Evaluate at the predicted point
            denoised_2 = model(x_2, sigma_down * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_down, denoised_2)

            # 3. Corrector step (average of gradients)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

            # Add noise according to ancestral sampling formula
            if sigma_up > 0:
                x = x + noise_sampler(sigma, sigmas[i + 1]) * s_noise * sigma_up

    return x


# Integration function for Automatic1111/Forge
@torch.no_grad()
def sample_heun_ancestral_integration(
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
    """Convenience function for Heun Ancestral sampler."""
    return sample_heun_ancestral(
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
