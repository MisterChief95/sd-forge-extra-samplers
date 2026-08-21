import logging
import math
from enum import Enum

import torch

from backend.logging import setup_logger

from modules import sd_samplers, sd_samplers_common
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules_forge.packages.k_diffusion.sampling import (
    get_ancestral_step,
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
    to_d,
)

logger = logging.getLogger("extra_samplers")
setup_logger(logger)


def clamp(x: int | float, lower: int | float, upper: int | float) -> int | float:
    return max(lower, min(x, upper))


# From ComfyUI
def default_noise_sampler(x, seed=None):
    """
    Default noise sampler for the extended reverse SDE solver.
    Generates Gaussian noise based on the input tensor's shape and device.
    If a seed is provided, it uses that seed for reproducibility.
    """
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(
        x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator
    )


class _Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args
        self.init_latent, self.mask, self.nmask = model.init_latent, model.mask, model.nmask

    def __enter__(self):
        # init_latent shares x's rank, so it resizes over every non-batch/channel axis;
        # mask and nmask are always 2D spatial maps, so they only take the trailing two.
        if self.init_latent is not None:
            self.model.init_latent = torch.nn.functional.interpolate(
                input=self.init_latent, size=self.x.shape[2:], mode=self.mode
            )
        if self.mask is not None:
            self.model.mask = torch.nn.functional.interpolate(
                input=self.mask.unsqueeze(0), size=self.x.shape[-2:], mode=self.mode
            ).squeeze(0)
        if self.nmask is not None:
            self.model.nmask = torch.nn.functional.interpolate(
                input=self.nmask.unsqueeze(0), size=self.x.shape[-2:], mode=self.mode
            ).squeeze(0)

        return self

    def __exit__(self, type, value, traceback):
        del self.model.init_latent, self.model.mask, self.model.nmask
        self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask


def _lattice_dims(x: torch.Tensor):
    """
    Positive indices of the two trailing spatial axes, plus the leading shape.

    Latents are normally [B, C, H, W], but Wan-VAE models (Qwen-Image, Krea, Anima)
    carry a singleton temporal axis - [B, C, 1, H, W]. Hardcoding dims 2 and 3 then
    lands on that T=1 axis, where the odd-size branch slices it down to 0 and unfold
    dies with "maximum size for tensor at dimension 2 is 0". Addressing H/W from the
    end works for both ranks, and leaving the rank otherwise intact keeps the nested
    model(...) call receiving the shape the model expects.
    """
    return x.ndim - 2, x.ndim - 1, x.shape[: x.ndim - 2]


def _lattice_gather(x, hd, wd, lead, m, n):
    """Split into 2x2 blocks and take each block's [1, 1] corner as a half-res latent."""
    a_list = x.unfold(hd, 2, 2).unfold(wd, 2, 2).contiguous().view(*lead, m * n, 2, 2)
    return a_list, a_list[..., 1, 1].view(*lead, m, n)


def _lattice_scatter(a_list, values, lead, m, n):
    """Write values back into each block's [1, 1] corner and fold back to full res."""
    a_list[..., 1, 1] = values.reshape(*lead, m * n)
    nlead = len(lead)
    perm = list(range(nlead)) + [nlead, nlead + 2, nlead + 1, nlead + 3]
    return a_list.view(*lead, m, n, 2, 2).permute(*perm).reshape(*lead, 2 * m, 2 * n)


def _restore_extras(x, original_shape, m, n, extra_row, extra_col, row_content, col_content):
    if not (extra_row or extra_col):
        return x

    x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
    x_expanded[..., : 2 * m, : 2 * n] = x

    if extra_row:
        x_expanded[..., -1:, : 2 * n + 1] = row_content

    if extra_col:
        x_expanded[..., : 2 * m, -1:] = col_content

    if extra_row and extra_col:
        x_expanded[..., -1:, -1:] = col_content[..., -1:, :]

    return x_expanded


@torch.no_grad()
def smea_sampling_step(x, model, dt, sigma_hat, **extra_args):
    # Scale only the trailing two axes; a 5D Wan-VAE latent's temporal axis stays at 1.
    spatial = x.shape[2:]
    scale_factor = (1.0,) * (len(spatial) - 2) + (1.25, 1.25)
    x = torch.nn.functional.interpolate(input=x, scale_factor=scale_factor, mode="nearest-exact")

    with _Rescaler(model, x, "nearest-exact", **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)

    d = to_d(x, sigma_hat, denoised)
    x = x + d * dt
    x = torch.nn.functional.interpolate(input=x, size=spatial, mode="nearest-exact")

    return x


@torch.no_grad()
def dy_sampling_step(x, model, dt, sigma_hat, **extra_args):
    """
    Take a Euler step on the half-res lattice of 2x2 block corners, leaving the other
    three quarters of the latent untouched.

    Note that what gets scattered back is the *stepped* c, not `denoised`. Writing the
    raw x0 prediction there instead drops those pixels straight to sigma=0 while their
    neighbours stay at sigma_hat, which stamps a Nyquist-frequency checkerboard into the
    latent - the model reads that as high-frequency garbage and the sample collapses.
    """
    original_shape = x.shape
    hd, wd, lead = _lattice_dims(x)
    m, n = original_shape[hd] // 2, original_shape[wd] // 2
    extra_row = original_shape[hd] % 2 == 1
    extra_col = original_shape[wd] % 2 == 1
    extra_row_content = extra_col_content = None

    if extra_row:
        extra_row_content = x[..., -1:, :]
        x = x[..., :-1, :]
    if extra_col:
        extra_col_content = x[..., :, -1:]
        x = x[..., :, :-1]

    a_list, c = _lattice_gather(x, hd, wd, lead, m, n)

    with _Rescaler(model, c, "nearest-exact", **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)

    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    x = _lattice_scatter(a_list, c, lead, m, n)

    return _restore_extras(x, original_shape, m, n, extra_row, extra_col, extra_row_content, extra_col_content)


def sampler_metadata(name: str, extra_params: dict = {}, sampler_aliases: list[str] = []):
    def decorator(func):
        func.sampler_extra_params = extra_params
        func.sampler_name = name
        func.sampler_k_names = [name.replace(" ", "_").lower(), *sampler_aliases]
        return func

    return decorator


def scheduler_metadata(name: str, alias: str, need_inner_model: bool = False):
    def decorator(func):
        func.name = name
        func.alias = alias
        func.need_inner_model = need_inner_model
        return func

    return decorator


class Interpolator(Enum):
    LINEAR = (lambda x: x,)  # noqa: E731
    COSINE = (lambda x: torch.sin(x * math.pi / 2),)  # noqa: E731
    SINE = (lambda x: 1 - torch.cos(x * math.pi / 2),)  # noqa: E731


# Original Implementation `ExtendIntermediateSigmas` by catboxanon: https://www.github.com/catboxanon/
# Original class impl: https://github.com/comfyanonymous/ComfyUI/blob/065d855f14968406051a1340e3f2f26461a00e5d/comfy_extras/nodes_custom_sampler.py#L253
def extend_sigmas(
    sigmas: torch.Tensor,
    steps: int,
    start_at_sigma: float,
    end_at_sigma: float,
    interpolator: Interpolator = Interpolator.LINEAR,
) -> torch.FloatTensor:
    if start_at_sigma < 0:
        start_at_sigma = float("inf")

    # linear space for our interpolation function
    x = torch.linspace(0, 1, steps + 1, device=sigmas.device)[1:-1]
    computed_spacing: torch.Tensor = interpolator.value[0](x)

    extended_sigmas: list[torch.Tensor] = []
    for i in range(len(sigmas) - 1):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i + 1]

        extended_sigmas.append(sigma_current)

        if end_at_sigma <= sigma_current <= start_at_sigma:
            interpolated_steps: torch.Tensor = computed_spacing * (sigma_next - sigma_current) + sigma_current
            extended_sigmas.extend(interpolated_steps.tolist())

    # Add the last sigma value
    if len(sigmas) > 0:
        extended_sigmas.append(sigmas[-1])

    return torch.FloatTensor(extended_sigmas).to(sigmas.device)


def ei_h_phi_1(h: torch.Tensor) -> torch.Tensor:
    """Compute h*phi_1(h) = expm1(h) for exponential integrator"""
    return torch.expm1(h)


def ei_h_phi_2(h: torch.Tensor) -> torch.Tensor:
    """Compute h*phi_2(h) = (expm1(h) - h) / h"""
    return (torch.expm1(h) - h) / h


def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable sqrt - clamps negative values before sqrt"""
    return (x.clamp(min=0) + eps).sqrt()


def is_rf_model(model) -> bool:
    """Detect rectified-flow / flow-matching models (Flux, SD3, etc.)."""
    predictor = getattr(getattr(model, "inner_model", None), "predictor", None)
    return getattr(predictor, "prediction_type", None) == "const"


def rf_churn_step(x: torch.Tensor, sigma: torch.Tensor, sigma_hat: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """
    Karras churn step (raising noise level from sigma to sigma_hat) for rectified-flow
    models, where alpha_t = 1 - sigma_t instead of the fixed alpha=1 assumed by the
    eps-parameterization formula `x - eps * sqrt(sigma_hat**2 - sigma**2)`.

    Since RF mixes signal and noise via a sigma-dependent alpha, raising the noise
    level also requires rescaling the signal component - the same alpha_ratio/
    renoise_coeff decomposition k-diffusion's sample_euler_ancestral_RF uses for its
    analogous ancestral noise-injection step, applied here in the churn direction
    (sigma -> sigma_hat, sigma_hat > sigma) instead of the ancestral one (sigma -> a
    lower sigma_down).

    RF's valid domain is sigma in [0, 1] (alpha = 1 - sigma), so sigma_hat is clamped
    below 1 to keep (1 - sigma_hat) positive. Note that this clamp is only ever a
    band-aid: an RF churn that pushes sigma_hat near 1 shrinks alpha_ratio toward 0 and
    thereby erases x almost entirely, so callers must keep the churn small rather than
    rely on the clamp - see churn_gamma.
    """
    sigma_hat = sigma_hat.clamp(max=1.0 - 1e-4)
    alpha_ratio = (1 - sigma_hat) / (1 - sigma).clamp(min=1e-4)
    sigma_up = safe_sqrt(sigma_hat**2 - sigma**2 * alpha_ratio**2)
    return alpha_ratio * x + eps * sigma_up


def rf_ancestral_step(sigma, sigma_next, eta: float = 1.0):
    """
    Ancestral step parameters for rectified-flow models.

    Returns (sigma_down, sigma_up, alpha_ratio). The caller steps deterministically to
    sigma_down, then renoises with `x = alpha_ratio * x + noise * s_noise * sigma_up`.

    k-diffusion's get_ancestral_step assumes alpha = 1, so it lands the *noise* variance
    on sigma_next correctly but leaves the signal at (1 - sigma_down) when RF needs
    (1 - sigma_next). At eta=1 that overweights the signal by ~1.96x on the first step
    and compounds to ~25x across 30 steps. alpha_ratio is the missing correction; it is
    exactly the decomposition sample_euler_ancestral_RF uses inline.

    Mirrors rf_churn_step, but in the descending direction (sigma -> a lower sigma_down).
    """
    if not eta:
        return sigma_next, sigma_next.new_zeros(()), sigma_next.new_ones(())

    # Scale the whole step down by eta rather than solving for sigma_up first: it keeps
    # sigma_down in [0, sigma_next] for any eta without a min() guard, and eta=1 recovers
    # the fully-ancestral sigma_down = sigma_next * (sigma_next / sigma).
    downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
    sigma_down = sigma_next * downstep_ratio

    alpha_next = 1 - sigma_next
    alpha_down = 1 - sigma_down
    alpha_ratio = alpha_next / alpha_down.clamp(min=1e-4)

    sigma_up = safe_sqrt(sigma_next**2 - sigma_down**2 * alpha_ratio**2)
    return sigma_down, sigma_up, alpha_ratio


def ancestral_step(model, sigma, sigma_next, eta: float = 1.0):
    """
    Ancestral step parameters, dispatched on model type.

    Returns (sigma_down, sigma_up, alpha_ratio), where alpha_ratio is None for eps models
    (no signal rescale needed, alpha is 1). Callers step deterministically to sigma_down,
    then:

        if alpha_ratio is not None:
            x = alpha_ratio * x
        if sigma_up > 0:
            x = x + noise * s_noise * sigma_up

    Applying alpha_ratio whenever it is not None - including when s_noise is 0 - matches
    sample_euler_ancestral_RF, which gates the rescale on eta rather than on s_noise.
    """
    if is_rf_model(model):
        return rf_ancestral_step(sigma, sigma_next, eta)

    sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=eta)
    return sigma_down, sigma_up, None


RF_SUBSTEP_START_SIGMA = 0.5
"""Rectified-flow sigma below which composition has settled enough for resolution substeps."""


def substep_schedule(model, sigmas, eps_indices, rf_start_sigma: float = RF_SUBSTEP_START_SIGMA):
    """
    Ordered step indices at which a resolution-changing substep should run.

    The SMEA/Dy samplers schedule their half-resolution and 1.25x-resolution substeps by
    step index, which puts them at sigma ~1.0-0.93 - precisely the window that fixes
    composition. A UNet tolerates that, being convolutional and largely translation
    equivariant, so a resize mostly rescales content. A DiT encodes absolute position
    through RoPE, so handing it a 1.25x latent makes it compose for a 1.25x canvas: a
    waist-up prompt comes back full-body, because the model really is laying out a taller
    scene.

    On rectified flow the same NUMBER of substeps is kept, but moved below rf_start_sigma
    where composition has already settled, so they still add detail and variation without
    redefining the framing. eps models get the original index schedule verbatim.

    Returns a list in schedule order, so callers that run different substep kinds can slice
    it (e.g. SMEA first, then Dy) and keep their relative ordering on both families.
    """
    if not is_rf_model(model):
        return list(eps_indices)

    picked = []
    for i in range(len(sigmas) - 1):
        if float(sigmas[i]) < rf_start_sigma and float(sigmas[i + 1]) > 0:
            picked.append(i)
            if len(picked) >= len(eps_indices):
                break

    return picked


EPS_REFERENCE_SIGMA_MAX = 14.6
"""Sigma ceiling of the eps schedules these samplers' hardcoded thresholds were tuned against."""


def scale_sigma_threshold(model, threshold: float, sigmas) -> float:
    """
    Rescale a hardcoded sigma threshold onto the current schedule's range.

    Several samplers compare sigma against literals chosen for eps models, where sigma runs
    to about 14.6. Rectified-flow sigma tops out at 1.0, so those comparisons stop meaning
    what they did: a `sigma > 2.0` high-noise test never fires, and a `sigma < 1.0`
    low-noise test fires on every step. Scaling by the schedule's own sigma_max restores
    the fraction of the trajectory the threshold was meant to select.

    Returns the threshold untouched on eps models, so their behaviour is unchanged.
    """
    if not is_rf_model(model):
        return threshold

    return threshold * float(sigmas[0]) / EPS_REFERENCE_SIGMA_MAX


def alpha_for(model_sampling, sigma):
    """
    alpha_t implied by sigma: 1 for eps-parameterized models, 1 - sigma for rectified flow.

    Derived as sigma * exp(half-logSNR) rather than branching on prediction_type, so it
    tracks whatever the model's predictor reports.
    """
    return sigma * sigma_to_half_log_snr(sigma, model_sampling).exp()


def cfg_pp_ancestral_step(model_sampling, x, sigma, sigma_next, denoised, uncond_denoised, eta: float = 0.0):
    """
    One CFG++ Euler/ancestral step, correct for both eps and rectified-flow models.

    A direct port of k-diffusion's sample_euler_ancestral_cfg_pp. Three things separate it
    from the `x = denoised + to_d(x, sigma, uncond_denoised) * sigma_down` form these
    samplers used to carry, and all three are invisible on eps models because alpha == 1:

      - uncond_denoised is scaled by alpha_s *before* to_d. Without that, to_d returns the
        rectified-flow velocity (n - x0) where the step needs the noise (n) - so the update
        direction is simply wrong, which is what drove RF CFG++ output almost black.
      - denoised is scaled by alpha_t, being an x0-space quantity written into a latent
        that carries alpha_t * x0.
      - the ancestral split runs on SNR-normalized sigmas (sigma / alpha), with sigma_down
        converted back by alpha_t.

    Returns (x, alpha_t, sigma_up). Callers renoise with
    `x = x + alpha_t * noise * s_noise * sigma_up`.

    Callers must pass sigmas through offset_first_sigma_for_snr: alpha_s is 0 at sigma=1,
    which is exactly where a rectified-flow schedule starts, and sigma / alpha_s diverges.
    """
    # Final step: half-logSNR is infinite at sigma=0, so alpha_for would evaluate 0 * inf
    # and poison the latent with NaN. Landing on denoised is what every reference does.
    if sigma_next == 0:
        return denoised, sigma_next.new_ones(()), sigma_next.new_zeros(())

    alpha_s = alpha_for(model_sampling, sigma)
    alpha_t, sigma_up, sigma_down = cfg_pp_noise_params(model_sampling, sigma, sigma_next, eta)

    d = to_d(x, sigma, alpha_s * uncond_denoised)

    return alpha_t * denoised + sigma_down * d, alpha_t, sigma_up


def cfg_pp_noise_params(model_sampling, sigma, sigma_next, eta: float = 0.0):
    """
    The CFG++ ancestral split for one step: (alpha_t, sigma_up, sigma_down).

    Split out so higher-order CFG++ samplers can build their own x update yet still renoise
    on the same terms as cfg_pp_ancestral_step. A caller mixing this split with the one from
    ancestral_step() will step to one noise level and renoise as if at another - identical on
    eps, where both reduce to the same get_ancestral_step call, but badly divergent on RF.
    """
    # Same sigma=0 guard as cfg_pp_ancestral_step: half-logSNR is infinite there, so alpha_for
    # would evaluate 0 * inf. Callers treat sigma_down == 0 as the final denoising step.
    if sigma_next == 0:
        zero = sigma_next.new_zeros(())
        return sigma_next.new_ones(()), zero, zero

    alpha_s = alpha_for(model_sampling, sigma)
    alpha_t = alpha_for(model_sampling, sigma_next)

    sigma_down, sigma_up = get_ancestral_step(sigma / alpha_s, sigma_next / alpha_t, eta=eta)
    return alpha_t, sigma_up, alpha_t * sigma_down


def churn_gamma(s_churn: float, n_steps: int, sigma, s_tmin: float, s_tmax: float) -> float:
    """
    Karras churn factor for this step, matching k-diffusion's sample_euler.

    These samplers previously wrote `max(s_churn / n_steps, 2**0.5 - 1)` where
    k-diffusion writes `min(...)`. Under `max`, gamma could never fall below 0.414, so
    churn ran pinned at its ceiling on every step even at the default s_churn=0, and the
    s_churn slider did nothing at all until it exceeded n_steps * 0.414 (>12 at 30 steps)
    - the one regime where `max` and `min` agree. Restoring `min` makes s_churn=0 mean no
    churn and makes the slider monotonic across its whole range.

    That correction also matters far more for RF than for eps. Eps churn only *adds*
    noise (the signal keeps coefficient 1) on an unbounded sigma, so a pinned 1.414x was
    survivable. RF mixes via alpha = 1 - sigma on a bounded [0, 1] domain, so 1.414x at
    sigma=0.9 overshoots sigma_max, clamps to ~1.0, and drives alpha_ratio to ~0.001,
    replacing the latent with pure noise on exactly the early steps that set composition.
    """
    if s_churn <= 0 or not (s_tmin <= sigma <= s_tmax):
        return 0.0

    return min(s_churn / n_steps, 2**0.5 - 1)


def setup_cfg_pp(extra_args: dict, post_cfg_function) -> dict:
    """
    Setup CFG++ by injecting post-cfg function into model_options.

    This is the standardized method for enabling CFG++ across all samplers.
    It uses the model_options dict approach which is more portable and
    maintainable than directly accessing UnetPatcher.

    Args:
        extra_args: The extra_args dict passed to the sampler
        post_cfg_function: Callback function to capture uncond_denoised.
                          Should have signature: (args: dict) -> denoised

    Returns:
        Modified extra_args dict with CFG++ configuration

    Example:
        def post_cfg_function(args):
            nonlocal uncond_denoised
            uncond_denoised = args["uncond_denoised"]
            return args["denoised"]

        if cfg_pp:
            extra_args = setup_cfg_pp(extra_args, post_cfg_function)
    """
    from backend.patcher.base import set_model_options_post_cfg_function

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )
    return extra_args


# NOTE: an enable_cfg_pp_simple() helper used to live here, pairing a
# `model.need_last_noise_uncond = True` flag with reads of `model.last_noise_uncond`.
# Nothing in Forge Classic ever set that attribute - CFGDenoiser is a plain nn.Module
# without it - so every sampler using it raised AttributeError on its first step. Use
# setup_cfg_pp() with a post_cfg_function that captures uncond_denoised instead.


def register_unique(label: str, func, aliases=None, options=None) -> None:
    aliases = list(dict.fromkeys(aliases or []))
    options = options or {}

    existing_names = {s.name for s in sd_samplers.all_samplers}
    existing_aliases = {a for s in sd_samplers.all_samplers for a in (getattr(s, "aliases", []) or [])}

    if label in existing_names or any(a in existing_aliases for a in aliases):
        logger.warning(f"'{label}' already registered (or alias collision). Skipping.")
        return

    def ctor(m, f=func):
        return KDiffusionSampler(f, m)

    sdata = sd_samplers_common.SamplerData(label, ctor, aliases, options)

    sd_samplers.all_samplers.append(sdata)
    sd_samplers.all_samplers_map = {s.name: s for s in sd_samplers.all_samplers}

    if hasattr(sd_samplers, "set_samplers"):
        sd_samplers.set_samplers()

    logger.info(f"registered sampler: {label}")
