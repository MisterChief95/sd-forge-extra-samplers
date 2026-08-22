import torch
from functools import partial
from tqdm.auto import trange

from modules_forge.packages.k_diffusion.sampling import (
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
    to_d,
    default_noise_sampler,
)

from lib_es.utils import (
    is_rf_model,
    alpha_for,
    ancestral_step,
    sampler_metadata,
    setup_cfg_pp,
)

# ==============================================================================================
# Second order multistep solver from https://arxiv.org/pdf/2308.02157
#
# Adapted from ComfyUI's comfy/k_diffusion/sampling.py `res_multistep`:
#   https://github.com/comfyanonymous/ComfyUI  (GPL-3.0)
#
# Deviates from upstream in two respects.
#
# 1. Upstream is eps-only. It takes the exponential integrator's time variable as -log(sigma),
#    which is the half-logSNR only when alpha == 1. Rectified-flow models have alpha = 1 - sigma,
#    so that is wrong for them. Here the time variable comes from the model's own predictor via
#    sigma_to_half_log_snr, which reduces to -log(sigma) exactly when alpha == 1.
#
# 2. The eta > 0 (ancestral) branch is no longer an "ancestral split". Upstream - and the first
#    revision of this file - stepped the deterministic integrator to a synthetic sigma_down, then
#    rescaled and renoised back up to sigmas[i + 1]. The exponential integrator's step size was
#    then h = lambda(sigma_down) - lambda(sigma_i), a *ghost* step: at eta = 1 sigma_down sits far
#    below sigmas[i + 1], so h was much larger than the step actually being taken and the
#    second-order extrapolation term was weighted for a step that never happened. That
#    over-sharpens, flattens soft gradients and dulls colour. See _RES_SDE_DERIVATION below for
#    the replacement, which folds the stochasticity into the integrator itself the way
#    DPM-Solver++(2M) SDE does, on the real step h = lambda(sigmas[i + 1]) - lambda(sigmas[i]).
#
# Both changes are no-ops at eta == 0, where the eta > 0 branch is not taken at all, so the
# deterministic samplers are bit-for-bit unchanged.
# ==============================================================================================

_RES_SDE_DERIVATION = """
Reverse SDE in half-logSNR time, for a general schedule x_t = alpha_t x0 + sigma_t eps.

Write lambda = log(alpha/sigma), f = (log alpha)', g^2 = (sigma^2)' - 2 f sigma^2. Then

    g^2 = 2 sigma^2 [(log sigma)' - (log alpha)'] = -2 sigma^2 lambda'          (exact, any alpha)

The eta-parameterized reverse SDE is

    dx = [f x - (1 + eta) (g^2 / 2) grad log p_t(x)] dt + sqrt(eta) g dw
       = [f x + (1 + eta) (g^2 / 2 sigma^2) (x - alpha x0)] dt + sqrt(eta) g dw

Using g^2 / (2 sigma^2) = -lambda' and changing the independent variable to lambda:

    dx/dlambda = [dlog(sigma)/dlambda - eta] x + (1 + eta) alpha x0 + noise

a linear ODE whose homogeneous solution from lambda_s to lambda_t (h = lambda_t - lambda_s) is
(sigma_t / sigma_s) exp(-eta h). Variation of constants, with alpha_u = sigma_u exp(u) giving
(sigma_t / sigma_u) alpha_u = alpha_t exp(u - lambda_t), yields the exact solution

    x_t = (sigma_t / sigma_s) e^{-eta h} x_s
        + (1 + eta) alpha_t INT_0^h e^{-(1 + eta)(h - tau)} x0(lambda_s + tau) dtau
        + sigma_t sqrt(1 - e^{-2 eta h}) z,        z ~ N(0, I)

The noise variance comes from INT (sigma_t/sigma_u)^2 e^{-2 eta (lambda_t - u)} eta |g^2 dt|,
and |g^2 dt| = 2 sigma_u^2 dlambda, so every sigma_u cancels: the result is sigma_t^2 (1 -
e^{-2 eta h}) regardless of alpha. Nothing above assumed alpha == 1 - all three terms are
alpha-generic, and alpha_t enters only as the true alpha at sigmas[i + 1]. So DPM-Solver++(2M)
SDE's x-decay, data and noise coefficients are already correct for rectified flow as written;
they need the right lambda (which this file's lambda_fn supplies) and nothing else.

Sanity check - exact marginal preservation. Feed in x_s = alpha_s x0 + sigma_s eps with x0
constant. Since (sigma_t alpha_s / sigma_s) = alpha_t e^{-h}, the x0 coefficient collapses to
alpha_t [e^{-h} e^{-eta h} + 1 - e^{-(1 + eta) h}] = alpha_t, and the noise variance to
sigma_t^2 e^{-2 eta h} + sigma_t^2 (1 - e^{-2 eta h}) = sigma_t^2. Exact for every eta and every
alpha.

RES second-order coefficients under this step.

RES (arXiv 2308.02157) replaces x0(lambda_s + tau) by the linear interpolant through its two
known nodes: `denoised` at tau = 0 and `old_denoised` at tau = c2 h, where

    c2 = (lambda(sigmas[i - 1]) - lambda(sigmas[i])) / h                       (h is the REAL step)

so x0(lambda_s + tau) ~ denoised + (tau / (c2 h)) (old_denoised - denoised). Substituting
u = (1 + eta) tau maps the integration range [0, h] onto [0, h_eta] with h_eta = (1 + eta) h,
and turns the weight into e^{-(h_eta - u)} du. Then

    I0 = INT_0^h w(tau) dtau      = 1 - e^{-h_eta}      = h_eta phi1(-h_eta)
    I1 = INT_0^h tau w(tau) dtau  = h_eta^2 phi2(-h_eta) / (1 + eta) = h_eta h phi2(-h_eta)

so the old_denoised weight is I1 / (c2 h) = h_eta phi2(-h_eta) / c2 and the whole step is

    x = (sigma_t/sigma_s) e^{-eta h} x + alpha_t h_eta (b1 denoised + b2 old_denoised) + noise
    b1 = phi1(-h_eta) - phi2(-h_eta) / c2
    b2 = phi2(-h_eta) / c2

i.e. structurally identical to the deterministic RES step with h -> h_eta in the phi functions
and in the prefactor - but c2 stays on the real-h axis. That is not an arbitrary choice: the same
substitution that carries tau -> u carries the node at c2 h to c2 h_eta, so c2 is the same
*fraction* of the interval in either variable. Measuring it against a ghost sigma_down, or
rescaling it by (1 + eta), would move the interpolation node off where old_denoised was actually
evaluated.

At eta = 0: h_eta = h, e^{-eta h} = 1, the noise scale is 0, and the expression is exactly the
deterministic RES step with sigma_down = sigmas[i + 1]. When old_denoised == denoised,
h_eta (b1 + b2) = h_eta phi1(-h_eta) = 1 - e^{-h_eta} and the step collapses to its first-order
form. On an eps model at eta = 1 that first-order form is *identical* to classic Euler-ancestral:
e^{-h} = sigma_t/sigma_s there, so the x coefficient is sigma_t^2/sigma_s^2 = sigma_down/sigma_s
and the noise scale is sigma_t sqrt(1 - (sigma_t/sigma_s)^2) = get_ancestral_step's sigma_up.
"""


def phi1_fn(t):
    """
    Computes the function phi1(t) = (exp(t) - 1) / t using PyTorch's expm1 function.
    Args:
        t (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: The result of (exp(t) - 1) / t.
    """

    return torch.expm1(t) / t


def phi2_fn(t):
    """
    Compute the value of the phi2 function.
    The phi2 function is defined as (phi1_fn(t) - 1.0) / t, where phi1_fn is
    another function that takes a single argument t.
    Parameters:
    t (float): The input value for the function.
    Returns:
    float: The computed value of the phi2 function.
    """

    return (phi1_fn(t) - 1.0) / t


@torch.no_grad()
def res_multistep(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    eta=1.0,
    cfg_pp=False,
):
    """
    Perform multi-step denoising using a conditional denoising model.
    Args:
        model (CFGDenoiserKDiffusion): The denoising model to use.
        x (torch.Tensor): The input tensor to be denoised.
        sigmas (list or torch.Tensor): A list or tensor of sigma values for each step.
        extra_args (dict, optional): Additional arguments to pass to the model. Defaults to None.
        callback (callable, optional): A callback function to be called after each step. Defaults to None.
        disable (bool, optional): If True, disables the progress bar. Defaults to None.
        s_noise (float, optional): Noise scale for stochasticity. Defaults to 1.0.
        noise_sampler (callable, optional): Function to sample noise. Defaults to None.
        eta (float, optional): Stochasticity of the reverse SDE. 0 solves the probability-flow
            ODE (deterministic); > 0 folds noise into the exponential integrator via
            h_eta = (1 + eta) * h and injects the matching variance. Defaults to 1.0.
        cfg_pp (bool, optional): If True, enables post-processing for classifier-free guidance. Defaults to False.
    Returns:
        torch.Tensor: The denoised output tensor.
    """
    extra_args = {} if extra_args is None else extra_args

    if cfg_pp and is_rf_model(model):
        raise RuntimeError(
            "Res Multistep CFG++ is not supported on rectified-flow models "
            "(Flux, SD3, Qwen-Image, Krea, Anima, Wan).\n"
            "\n"
            "This sampler's second-order CFG++ step has no published rectified-flow form, "
            "and the generalizations tried here produced pure noise rather than an image. "
            "Rather than emit a broken result it stops here.\n"
            "\n"
            "Use one of these instead, which are verified on rectified flow:\n"
            "  - Euler a CFG++            (built in)\n"
            "  - Euler a Multipass CFG++\n"
            "  - Gradient Estimation CFG++\n"
            "  - Res Multistep Ancestral  (no CFG++; second-order and rectified-flow correct)\n"
            "\n"
            "Res Multistep CFG++ remains fully supported on epsilon models (SD1.5, SDXL)."
        )

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    uncond_denoised = None

    # unconditional denoised is used for the second order multistep method
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        extra_args = setup_cfg_pp(extra_args, post_cfg_function)

    # The exponential integrator below runs in half-logSNR space, lambda = log(alpha/sigma).
    # This used to hardcode -log(sigma), which is only lambda when alpha == 1; rectified-flow
    # models have alpha = 1 - sigma, so lambda is logit(sigma).neg() instead. Deriving it via
    # the model's predictor keeps both families correct - the same approach seeds_2/seeds_3
    # and sa_solver already take. offset_first_sigma_for_snr nudges a leading sigma of exactly
    # 1.0 off the boundary, where RF's logit would otherwise be infinite.
    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    # eta > 0 takes the SDE-native construction; eta == 0 keeps the deterministic split path
    # verbatim (the two coincide there anyway, but running the same code guarantees it bitwise).
    # CFG++ is excluded: see the comment at the head of the deterministic branch below.
    sde_branch = eta > 0 and not cfg_pp

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sde_branch:
            # ------------------------------------------------------------------------------
            # RES on the eta-parameterized reverse SDE. Full derivation in
            # _RES_SDE_DERIVATION above; the short version is that h is the REAL step to
            # sigmas[i + 1] (never a ghost sigma_down), the stochasticity rides inside the
            # integrator via h_eta = (1 + eta) h, and the injected noise is exactly the
            # variance that leaves the marginal at sigmas[i + 1] intact.
            # ------------------------------------------------------------------------------
            if sigmas[i + 1] == 0:
                # half-logSNR is infinite at sigma=0, so h and alpha_t are unusable there.
                x = denoised
            else:
                lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
                h = lambda_t - lambda_s
                h_eta = h * (eta + 1)

                # Identical to DPM-Solver++(2M) SDE's `sigmas[i + 1] * lambda_t.exp()`;
                # alpha_for is that same product, routed through the model's predictor.
                alpha_t = alpha_for(model_sampling, sigmas[i + 1])
                x_decay = (sigmas[i + 1] / sigmas[i]) * (-h * eta).exp()

                if old_denoised is None:
                    # First step: only one x0 node exists, so the interpolant is constant and
                    # h_eta * phi1(-h_eta) = 1 - exp(-h_eta) is the whole data weight.
                    x = x_decay * x + alpha_t * (-h_eta).expm1().neg() * denoised
                else:
                    # c2 is measured on the real-h axis, between the lambda where old_denoised
                    # was evaluated and the lambda x carries now. It is a *fraction* of the
                    # step, so it is invariant under the tau -> (1 + eta) tau substitution that
                    # produces h_eta; only the phi arguments and the prefactor pick up eta.
                    c2 = (lambda_fn(sigmas[i - 1]) - lambda_s) / h

                    phi1_val, phi2_val = phi1_fn(-h_eta), phi2_fn(-h_eta)
                    b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
                    b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

                    x = x_decay * x + alpha_t * h_eta * (b1 * denoised + b2 * old_denoised)

                if s_noise > 0:
                    noise_scale = sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()
                    x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * noise_scale * s_noise

            old_denoised = denoised
            continue

        # ----------------------------------------------------------------------------------
        # eta == 0, or CFG++ at any eta.
        #
        # CFG++ deliberately stays on the ancestral split. Its update is not a plain
        # exponential-integrator step: the `x = x + (denoised - uncond_denoised)` correction is
        # defined relative to a deterministic substep landing on sigma_down, and there is no
        # published CFG++ analogue of the h_eta construction to port it onto. It is also
        # reachable only on eps models (the RF case raises above), where alpha == 1 and the
        # split is at least self-consistent. Rewriting it would be an unverifiable redesign, so
        # the upstream-equivalent form is preserved exactly as it was.
        # ----------------------------------------------------------------------------------
        sigma_down, sigma_up, alpha_ratio = ancestral_step(model, sigmas[i], sigmas[i + 1], eta)

        if sigma_down == 0 or old_denoised is None:
            # Euler method
            if cfg_pp:
                d = to_d(x, sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                d = to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
        else:
            # Second order multistep method in https://arxiv.org/pdf/2308.02157
            lambda_s, lambda_next = lambda_fn(sigmas[i]), lambda_fn(sigma_down)
            lambda_prev = lambda_fn(sigmas[i - 1])
            h = lambda_next - lambda_s

            # c2 locates old_denoised on this step's local time axis, so it has to be measured
            # between the lambda where old_denoised was evaluated (sigmas[i - 1]) and the lambda
            # x actually carries now (sigmas[i]). Upstream measures from the previous step's
            # sigma_down, which is only where the previous *deterministic* substep landed before
            # the ancestral noise put x back up at sigmas[i]. At eta == 0 the two are the same
            # point, so this is a no-op there; at eta > 0 sigma_down sits well below sigmas[i]
            # (roughly |c2| twice too large at eta=1), which shrinks the b2 term that extrapolates
            # the drift of denoised between steps. That shortfall has a fixed sign and a different
            # size per latent channel, so it accumulates into the colour tint the ancestral variant
            # showed. DPM-Solver++(2M) SDE takes the same nominal-schedule gap at any eta.
            c2 = (lambda_prev - lambda_s) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            # DPM-Solver++ data prediction: x carries at sigma_down/sigma_s, and the data
            # terms are weighted by alpha at the destination. For alpha == 1 this reduces to
            # the previous exp(-h) * x + h * (...) form exactly.
            sigma_ratio = sigma_down / sigmas[i]
            alpha_down = alpha_for(model_sampling, sigma_down)

            # Only eps reaches this with cfg_pp set; RF + cfg_pp raises at the top of the call.
            if cfg_pp:
                x = x + (denoised - uncond_denoised)
                x = sigma_ratio * x + alpha_down * h * (b1 * uncond_denoised + b2 * old_denoised)
            else:
                x = sigma_ratio * x + alpha_down * h * (b1 * denoised + b2 * old_denoised)

        # Rescale the signal (rectified flow only), then add the ancestral noise
        if alpha_ratio is not None:
            x = alpha_ratio * x

        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        old_denoised = uncond_denoised if cfg_pp else denoised
    return x


@sampler_metadata(
    "Res Multistep",
    {"scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep(
    model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=0.0,
        cfg_pp=False,
    )


@sampler_metadata(
    "Res Multistep CFG++",
    {"scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep_cfg_pp(
    model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=0.0,
        cfg_pp=True,
    )


@sampler_metadata(
    "Res Multistep Ancestral",
    {"uses_ensd": True, "scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep_ancestral(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=eta,
        cfg_pp=False,
    )


@sampler_metadata(
    "Res Multistep Ancestral CFG++",
    {"uses_ensd": True, "scheduler": "sgm_uniform"},
)
@torch.no_grad()
def sample_res_multistep_ancestral_cfg_pp(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None
):
    return res_multistep(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        eta=eta,
        cfg_pp=True,
    )
