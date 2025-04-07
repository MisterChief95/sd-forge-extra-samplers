import torch
from tqdm import trange

from k_diffusion.sampling import to_d


# From ComfyUI
@torch.no_grad()
def sample_gradient_estimation(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.0):
    """Gradient-estimation sampler. Paper: https://openreview.net/pdf?id=o2ND9v0CeK"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        dt = sigmas[i + 1] - sigmas[i]
        if i == 0:
            # Euler method
            x = x + d * dt
        else:
            # Gradient estimation
            d_bar = ge_gamma * d + (1 - ge_gamma) * old_d
            x = x + d_bar * dt
        old_d = d
    return x
