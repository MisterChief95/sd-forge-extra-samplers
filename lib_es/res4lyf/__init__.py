#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/__init__.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
import sys
import types

import torch
import torch.nn.functional as F


def _install_comfy_shim():
    if "comfy" in sys.modules:
        return

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []

    model_sampling = types.ModuleType("comfy.model_sampling")

    class EPS:
        pass

    class CONST:
        pass

    class V_PREDICTION:
        pass

    class EDM:
        pass

    class X0:
        pass

    class IMG_TO_IMG:
        pass

    model_sampling.EPS = EPS
    model_sampling.CONST = CONST
    model_sampling.V_PREDICTION = V_PREDICTION
    model_sampling.EDM = EDM
    model_sampling.X0 = X0
    model_sampling.IMG_TO_IMG = IMG_TO_IMG

    samplers = types.ModuleType("comfy.samplers")
    samplers.SCHEDULER_NAMES = [
        "normal",
        "karras",
        "exponential",
        "sgm_uniform",
        "simple",
        "ddim_uniform",
    ]

    def _unsupported_scheduler(*args, **kwargs):
        raise NotImplementedError("Comfy scheduler helpers are not available in the Forge RES4LYF port.")

    samplers.calculate_sigmas = _unsupported_scheduler
    samplers.beta_scheduler = _unsupported_scheduler
    samplers.linear_quadratic_schedule = _unsupported_scheduler

    utils = types.ModuleType("comfy.utils")

    def bislerp(samples, width, height):
        try:
            from backend.misc.image_resize import bislerp as forge_bislerp

            return forge_bislerp(samples, width, height)
        except Exception:
            return F.interpolate(samples, size=(height, width), mode="bilinear", align_corners=False)

    utils.bislerp = bislerp

    model_patcher = types.ModuleType("comfy.model_patcher")

    def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
        from backend.patcher.base import set_model_options_post_cfg_function as forge_set_post_cfg

        return forge_set_post_cfg(
            model_options,
            post_cfg_function,
            disable_cfg1_optimization=disable_cfg1_optimization,
        )

    model_patcher.set_model_options_post_cfg_function = set_model_options_post_cfg_function

    supported_models = types.ModuleType("comfy.supported_models")

    k_diffusion = types.ModuleType("comfy.k_diffusion")
    k_diffusion_sampling = types.ModuleType("comfy.k_diffusion.sampling")

    class BrownianTreeNoiseSampler:
        def __init__(self, x, sigma_min, sigma_max, seed=None, cpu=False):
            self.x = x
            self.generator = torch.Generator(device="cpu" if cpu else x.device)
            if seed is not None:
                self.generator.manual_seed(int(seed))

        def __call__(self, sigma, sigma_next):
            return torch.randn(
                self.x.shape,
                dtype=self.x.dtype,
                layout=self.x.layout,
                device=self.x.device,
                generator=self.generator if self.generator.device == self.x.device else None,
            )

    k_diffusion_sampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSampler
    k_diffusion.sampling = k_diffusion_sampling

    common_dit = types.ModuleType("comfy.ldm.common_dit")

    def pad_to_patch_size(x, patch_size):
        patch_h, patch_w = patch_size
        pad_h = (-x.shape[-2]) % patch_h
        pad_w = (-x.shape[-1]) % patch_w
        return F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x

    common_dit.pad_to_patch_size = pad_to_patch_size

    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    ldm.common_dit = common_dit

    comfy.model_sampling = model_sampling
    comfy.samplers = samplers
    comfy.utils = utils
    comfy.model_patcher = model_patcher
    comfy.supported_models = supported_models
    comfy.k_diffusion = k_diffusion
    comfy.ldm = ldm

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_sampling"] = model_sampling
    sys.modules["comfy.samplers"] = samplers
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy.model_patcher"] = model_patcher
    sys.modules["comfy.supported_models"] = supported_models
    sys.modules["comfy.k_diffusion"] = k_diffusion
    sys.modules["comfy.k_diffusion.sampling"] = k_diffusion_sampling
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.common_dit"] = common_dit


_install_comfy_shim()
