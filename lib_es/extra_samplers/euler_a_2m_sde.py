import torch
from tqdm.auto import trange

from backend.modules.k_diffusion_extra import default_noise_sampler
from k_diffusion.sampling import BrownianTreeNoiseSampler, to_d, get_ancestral_step
from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion


def sigma_fn(t):
    """
    Computes the sigma function for a given tensor `t`.
    The sigma function is defined as the exponential of the negation of `t`.
    """
    return t.neg().exp()


def t_fn(sigma):
    """
    Computes the negative logarithm of the input tensor.
    """
    return sigma.log().neg()


def phi1_fn(t):
    """
    Computes the function phi1(t) = (exp(t) - 1) / t using PyTorch's expm1 function.
    Safely handles edge cases to prevent NaN or Inf values.
    """
    # Handle edge cases where t is close to zero
    is_small = t.abs() < 1e-8
    result = torch.ones_like(t)
    nonzero_mask = ~is_small
    if nonzero_mask.any():
        result[nonzero_mask] = torch.expm1(t[nonzero_mask]) / t[nonzero_mask]
    return torch.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)


def safe_division(numerator, denominator, default_value=0.0):
    """
    Safely divide tensors, handling division by zero with a default value.
    """
    result = torch.zeros_like(numerator)
    mask = denominator != 0
    if mask.any():
        result[mask] = numerator[mask] / denominator[mask]
    result[~mask] = default_value
    return torch.nan_to_num(result, nan=default_value, posinf=default_value, neginf=default_value)


def euler_ancestral_2m_sde(
    model: CFGDenoiserKDiffusion,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    cfg_pp=False,
    solver_type="midpoint",
    use_brownian=True,
    noise_strength_schedule=None,
):
    """
    Improved Euler Ancestral 2M SDE sampler with enhanced stability and quality control.

    Args:
        model: The denoising model
        x: Initial noise tensor
        sigmas: Noise schedule
        extra_args: Additional arguments for the model
        callback: Callback function for visualization
        disable: Whether to disable progress bar
        eta: Controls the stochasticity (0 = deterministic, 1 = full stochasticity)
        s_noise: Noise scale parameter
        noise_sampler: Custom noise sampler
        cfg_pp: Whether to enable CFG++ post-processing
        solver_type: The solver method ('midpoint' or 'heun')
        use_brownian: Whether to use Brownian noise (True) or independent random noise (False)
        noise_strength_schedule: Optional function to modify noise strength at each step
    """
    extra_args = {} if extra_args is None else extra_args

    # Create appropriate noise sampler
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if use_brownian and noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
    elif noise_sampler is None:
        noise_sampler = default_noise_sampler(x)

    # Input conditioning scalar
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    uncond_denoised = None

    # Default noise strength schedule if none provided
    if noise_strength_schedule is None:
        # Gradually reduce noise impact in later steps for sharper results
        def noise_strength_schedule(i, num_steps):
            # Start at 100% of s_noise, gradually reduce to 80% at the end
            return s_noise * (1.0 - 0.2 * i / (num_steps - 1))

    # Set up CFG++ if requested
    if cfg_pp:
        try:
            model.need_last_noise_uncond = True
            unet_patcher = model.inner_model.inner_model.forge_objects.unet
            # Disable optimizations that might interfere with CFG++
            unet_patcher.model_options["disable_cfg1_optimization"] = True

            # Define a function to capture the unconditional denoised prediction
            def post_cfg_function(args):
                nonlocal uncond_denoised
                uncond_denoised = args["uncond_denoised"]
                return args["denoised"]

            unet_patcher.set_model_sampler_post_cfg_function(post_cfg_function)
        except Exception as e:
            print(f"Warning: CFG++ setup failed: {e}. Falling back to standard sampling.")
            cfg_pp = False

    num_steps = len(sigmas) - 1
    for i in trange(num_steps, disable=disable):
        # Current and next sigma values
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Calculate current noise strength
        current_noise_strength = noise_strength_schedule(i, num_steps)

        # Get the denoised prediction at current step
        try:
            denoised = model(x, sigma_cur * s_in, **extra_args)
            # Ensure no NaN values in denoised output
            if torch.isnan(denoised).any():
                print(f"Warning: NaN detected in denoised output at step {i}. Correcting...")
                denoised = torch.nan_to_num(denoised)
        except Exception as e:
            print(f"Error during denoising at step {i}: {e}")
            # If denoising fails, create a fallback by linear interpolation with previous step
            if old_denoised is not None:
                denoised = old_denoised
            else:
                # If no previous step, use the input with small noise
                denoised = x - sigma_cur * torch.randn_like(x) * 0.1

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_cur, "sigma_hat": sigma_cur, "denoised": denoised})

        # Implement ancestral sampling step with stochasticity controlled by eta
        sigma_down, sigma_up = get_ancestral_step(sigma_cur, sigma_next, eta=eta)

        # Handle edge cases: ensure sigma values are valid
        sigma_down = torch.max(sigma_down, torch.tensor(1e-8, device=sigma_down.device))

        if old_denoised is None or sigma_next == 0:
            # First step: Use standard Euler Ancestral
            if cfg_pp and uncond_denoised is not None:
                # Apply CFG++ correction
                try:
                    d = model.last_noise_uncond
                    x = denoised + d * sigma_down
                except Exception as e:
                    print(f"CFG++ application failed: {e}. Using standard update.")
                    d = to_d(x, sigma_cur, denoised)
                    dt = sigma_down - sigma_cur
                    x = x + d * dt
            else:
                # Standard first-order update
                d = to_d(x, sigma_cur, denoised)
                dt = sigma_down - sigma_cur
                x = x + d * dt
        else:
            # Second-order multistep update
            t, _ = t_fn(sigma_cur), t_fn(sigma_down)
            sigma_calc = sigma_down - sigma_cur

            # Apply multistep update based on chosen solver
            try:
                if solver_type == "heun":
                    # Heun's method style update (more accurate but slightly slower)
                    # First predict an intermediate point
                    d = to_d(x, sigma_cur, denoised)
                    x_2 = x + d * sigma_calc

                    # Get denoised at the intermediate point
                    denoised_2 = model(x_2, sigma_down * s_in, **extra_args)
                    if torch.isnan(denoised_2).any():
                        denoised_2 = torch.nan_to_num(denoised_2)

                    # Get derivative at intermediate point
                    d_2 = to_d(x_2, sigma_down, denoised_2)

                    # Final update with weighted average of derivatives
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * sigma_calc
                else:
                    # Midpoint method (faster but slightly less accurate)
                    # Calculate intermediate sigma value
                    t_mid = t + sigma_calc / 2  # Midpoint in t-space
                    sigma_mid = sigma_fn(t_mid)  # Convert back to sigma

                    # First get to the midpoint
                    d = to_d(x, sigma_cur, denoised)
                    x_mid = x + d * (sigma_mid - sigma_cur)  # Keep sigma diff for numerical stability

                    # Get denoised at midpoint
                    denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
                    if torch.isnan(denoised_mid).any():
                        denoised_mid = torch.nan_to_num(denoised_mid)

                    # Use midpoint derivative for full step
                    d_mid = to_d(x_mid, sigma_mid, denoised_mid)
                    x = x + d_mid * sigma_calc

                # Ensure no NaN values in x
                if torch.isnan(x).any():
                    print(f"Warning: NaN detected in latent at step {i}. Correcting...")
                    x = torch.nan_to_num(x)

                # Apply CFG++ correction if enabled
                if cfg_pp and uncond_denoised is not None:
                    x = x + (denoised - uncond_denoised)

            except Exception as e:
                print(f"Error in multistep update at step {i}: {e}")
                # Fallback to simple Euler step
                d = to_d(x, sigma_cur, denoised)
                dt = sigma_down - sigma_cur
                x = x + d * dt

        # Add SDE noise component controlled by sigma_up
        if sigma_up > 0:
            try:
                # Apply noise with current strength factor
                noise = noise_sampler(sigma_cur, sigma_next)
                # Scale noise according to the schedule
                x = x + noise * current_noise_strength * sigma_up

                # Ensure we don't have any NaN values after adding noise
                if torch.isnan(x).any():
                    print(f"Warning: NaN detected after adding noise at step {i}. Correcting...")
                    x = torch.nan_to_num(x)
            except Exception as e:
                print(f"Error adding noise at step {i}: {e}")

        # Store current denoised prediction for next step
        old_denoised = denoised

    # Final cleanup to ensure no NaN/Inf values
    x = torch.nan_to_num(x)
    return x


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        try:
            import torchsde

            # Handle edge cases for t0 and t1
            if t0 == t1:
                t1 = t0 + 1e-5

            t0, t1, self.sign = self.sort(t0, t1)
            w0 = kwargs.get("w0", torch.zeros_like(x))
            if seed is None:
                seed = torch.randint(0, 2**63 - 1, []).item()
            self.batched = True
            try:
                assert len(seed) == x.shape[0]
                w0 = w0[0]
            except (TypeError, AssertionError):
                seed = [seed]
                self.batched = False

            # Create brownian trees with error handling
            self.trees = []
            for s in seed:
                try:
                    tree = torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs)
                    self.trees.append(tree)
                except Exception as e:
                    print(f"Error creating individual tree: {e}")
                    # Create a dummy tree that returns random noise
                    self.trees.append(lambda t0, t1: torch.randn_like(w0))

            if not self.trees:
                raise ValueError("Failed to create any valid Brownian trees")

        except Exception as e:
            print(f"Error in BatchedBrownianTree initialization: {e}")
            # Create a fallback mechanism
            self.fallback = True
            self.shape = x.shape

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        # Check if we're using the fallback mechanism
        if hasattr(self, "fallback") and self.fallback:
            return torch.randn(self.shape)

        try:
            t0, t1, sign = self.sort(t0, t1)

            # Safety check - if times are too close, return small noise
            if torch.abs(t1 - t0) < 1e-10:
                if self.batched:
                    return torch.randn(len(self.trees), *self.trees[0](t0, t0).shape) * 1e-10
                else:
                    return torch.randn_like(self.trees[0](t0, t0)) * 1e-10

            w = torch.stack([torch.nan_to_num(tree(t0, t1)) for tree in self.trees]) * (self.sign * sign)
            return w if self.batched else w[0]
        except Exception as e:
            print(f"Error in Brownian sampling: {e}. Using random noise.")
            if self.batched:
                return torch.randn(len(self.trees), *self.trees[0](t0, t0).shape)
            else:
                return torch.randn_like(self.trees[0](t0, t0))


@torch.no_grad()
def sample_euler_ancestral_2m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    use_brownian=True,
):
    """
    Convenience function for the Euler Ancestral 2M SDE sampler without CFG++.

    Added control over Brownian noise vs random noise.
    """
    return euler_ancestral_2m_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        cfg_pp=False,
        solver_type=solver_type,
        use_brownian=use_brownian,
    )


@torch.no_grad()
def sample_euler_ancestral_2m_sde_cfgpp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    use_brownian=True,
):
    """
    Convenience function for the Euler Ancestral 2M SDE sampler with CFG++.

    Added control over Brownian noise vs random noise.
    """
    return euler_ancestral_2m_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        cfg_pp=True,
        solver_type=solver_type,
        use_brownian=use_brownian,
    )
