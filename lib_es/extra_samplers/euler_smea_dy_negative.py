import torch

from modules_forge.packages.k_diffusion.sampling import to_d

from tqdm.auto import trange

from lib_es.utils import dy_sampling_step, smea_sampling_step
from lib_es.utils import churn_gamma, is_rf_model, rf_churn_step, sampler_metadata


def _negative_substep_rebalance(x, x_pre, denoised, d, dt, sigma_next, rf: bool, flip_scale=1.0):
    """
    The sign flip that gives this sampler its name, in whichever parameterization the
    model uses.

    `x` is the substep's output, `x_pre` the latent as it entered the substep (i.e. the
    outer Euler result), `denoised` the outer model call's x0 prediction at sigma_hat.

    Eps branch (`-x - d * dt`, verbatim): with x = x0 + sigma * n and d = n, the Euler
    result is x0 + sigma_next * n, so the rebalance evaluates to

        -x0 + (sigma_hat - 2 * sigma_next) * n

    i.e. the x0 term is negated and the noise term is both re-leveled and sign-flipped.
    The negated x0 is harmless only because eps fires this at sigma_hat 14.6/12/10, where
    the noise term outweighs it several times over and the following steps rewrite it -
    what actually survives the operation is a sign-flipped resample of the noise.

    RF branch. The same expression on x = (1 - sigma) * x0 + sigma * n evaluates to

        -(1 + sigma_hat - 2 * sigma_next) * x0 + (sigma_hat - 2 * sigma_next) * n

    whose leading coefficient is negative across the whole schedule, so on RF - where the
    x0 term carries the alpha = 1 - sigma signal rather than being negligible - the flip
    lands as a literally inverted image. What it needs to reproduce is not that arithmetic
    but what the eps flip *accomplishes*: negate the noise residual, leave the predicted
    image alone.

    Doing that requires knowing which part of the latent is noise. It is recoverable in
    closed form from the outer step, with no extra model call. Writing N for the noise
    implied by the outer prediction, x = (1 - sigma_hat) * denoised + sigma_hat * N, the
    Euler update d = to_d(x, sigma_hat, denoised) = N - denoised gives

        x_pre = x + d * dt = (1 - sigma_next) * denoised + sigma_next * N

    exactly - an identity, not an approximation, since N is *defined* by that first line.
    So sigma_next * N = x_pre - (1 - sigma_next) * denoised, and the flip is

        x - 2 * flip_scale * (x_pre - (1 - sigma_next) * denoised)

    which subtracts twice the noise content and touches nothing else.

    Why not the previous `2 * (1 - sigma_next) * denoised - x`: that is this formula's
    special case for x == x_pre, i.e. for a substep that did nothing. Both substeps
    take a *second* Euler step of size dt on top of the one already taken, from a model
    call made at a different resolution. Writing beta = dt / sigma_hat and D' for that
    inner call's prediction, the substep output decomposes as

        x = (1 + beta) * (1 - sigma_next) * denoised - beta * D' + tau * N,
        tau = (1 + beta) * sigma_next = sigma_next**2 / sigma_hat

    (verified numerically against the real utils functions). Reflecting that through the
    *pre-substep* signal (1 - sigma_next) * denoised produces a signal coefficient of
    1 - 2 * sigma_next + tau where the latent's own is 1 - tau - an ~11% signal
    attenuation per flip at the sigma ~0.5 where RF substeps run - and simultaneously
    negates -beta * D', the substep's entire contribution. Both errors pull the image
    toward a smaller, blander signal: that is the desaturation/wash-out the reflection
    traded the colour inversion for. Subtracting the noise term directly avoids touching
    the signal at all, so whatever the substep found survives at full strength.

    flip_scale accounts for the substep landing at tau rather than sigma_next:

      - SMEA resizes the whole latent, so every pixel carries tau * N and the caller
        passes flip_scale = sigma_next / sigma_hat (tau / sigma_next). Exact whenever the
        1.25x up/downsample round-trips, which nearest-exact does for every latent size
        with an integral 1.25x (all multiples of 4, so every standard resolution).
      - Dy only rewrites the [1, 1] corner of each 2x2 block; the other three quarters are
        still x_pre at sigma_next, so flip_scale stays 1.0. The corners are then
        over-flipped by 2 * (sigma_next - tau) * N, which is |dt| / sigma_hat of the noise
        on a quarter of the latent - about 1% of the latent, versus the ~25% signal error
        the alternative would incur on the other three quarters.

    Note the flip is applied at sigma_next, not sigma_hat: the latent being flipped has
    already taken its Euler step. Reflecting a sigma_hat-level noise term out of it would
    over-subtract by the full step size, and the `- d * dt` re-leveling the eps branch
    carries is likewise dropped - on eps it only moves the noise scale (alpha is fixed at
    1), but on RF the noise level and the signal weight are one parameter, so landing off
    schedule is a signal-scale error rather than a bigger jump.
    """
    if rf:
        return x - 2 * flip_scale * (x_pre - (1 - sigma_next) * denoised)

    return -x - d * dt


@sampler_metadata("Euler SMEA Dy Negative")
@torch.no_grad()
def sample_euler_smea_dy_negative(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    rf = is_rf_model(model)

    # SMEA (step 0) then Dy (steps 1, 2), always at the original indices - deliberately
    # NOT relocated via substep_schedule for rectified flow, matching euler_smea.py and
    # euler_smea_dy.py.
    #
    # The reflection fix above makes relocation possible (it preserves the signal at any
    # sigma, unlike the old whole-latent negation, which only tolerated running near
    # sigma ~1.0 where there was barely any signal to invert). But running the substeps at
    # sigma ~1.0 lets them dictate framing - a DiT/RoPE model can reframe the whole scene
    # around a resized latent while composition is still forming (e.g. a waist-up prompt
    # coming back full-body) - and side-by-side testing showed that composition drift is
    # exactly where this sampler's extra variety comes from. Kept as a deliberate tradeoff,
    # not an oversight.
    # TEMP: matching upstream's literal `i + 1 // 2 == 1` precedence bug for an A/B test.
    # `//` binds tighter than `+`, so that's `i + 0 == 1` -> i == 1 only (Dy fires once,
    # not twice). Revert to the commented-out {1} version below to test that side again.
    smea_steps = {0}
    dy_steps = {1, 2}
    # smea_steps = {0}
    # dy_steps = {1}

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = churn_gamma(s_churn, len(sigmas) - 1, sigmas[i], s_tmin, s_tmax)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0 and rf:
            sigma_hat = sigma_hat.clamp(max=1.0 - 1e-4)
        dt = sigmas[i + 1] - sigma_hat

        if gamma > 0:
            if rf:
                x = rf_churn_step(x, sigmas[i], sigma_hat, eps)
            else:
                x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        # Euler method
        x = x + d * dt

        if sigmas[i + 1] > 0:
            if i in dy_steps:
                x_pre = x
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
                # Three quarters of the latent come back untouched at sigma_next, so the
                # noise to negate is the one the outer step left there: flip_scale 1.
                x = _negative_substep_rebalance(x, x_pre, denoised, d, dt, sigmas[i + 1], rf)

            if i in smea_steps:
                x_pre = x
                x = smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
                # SMEA re-steps the entire latent, leaving it at tau = sigma_next**2 /
                # sigma_hat, so the noise term to negate is scaled by tau / sigma_next.
                x = _negative_substep_rebalance(
                    x, x_pre, denoised, d, dt, sigmas[i + 1], rf, flip_scale=sigmas[i + 1] / sigma_hat
                )

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

    return x
