#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/sigmas.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
import copy

import torch

from modules import sd_schedulers
from modules_forge.packages.k_diffusion.external import ForgeScheduleLinker


def get_sigmas(model, scheduler, steps, denoise, shift=0.0, lq_threshold_noise=0.025):
    inner_model = getattr(model, "inner_model", None)

    if shift > 1e-6:
        predictor = getattr(inner_model, "predictor", None)
        if predictor is not None and hasattr(predictor, "set_parameters"):
            predictor = copy.deepcopy(predictor)
            predictor.set_parameters(shift=shift)
            inner_model = ForgeScheduleLinker(predictor)

    scheduler_entry = sd_schedulers.schedulers_map.get(scheduler) if scheduler else None
    if scheduler_entry is not None and scheduler_entry.function is not None and hasattr(inner_model, "sigmas"):
        kwargs = {"sigma_min": inner_model.sigmas[0].item(), "sigma_max": inner_model.sigmas[-1].item()}
        if scheduler_entry.need_inner_model:
            kwargs["inner_model"] = inner_model
        if scheduler_entry.name == "linear_quadratic":
            kwargs["threshold_noise"] = lq_threshold_noise
        sigmas = scheduler_entry.function(n=steps, **kwargs, device="cpu")
    elif inner_model is not None and hasattr(inner_model, "get_sigmas"):
        sigmas = inner_model.get_sigmas(steps)
    elif hasattr(model, "get_sigmas"):
        sigmas = model.get_sigmas(steps)
    else:
        raise NotImplementedError("RES4LYF Forge port could not resolve a sigma scheduler for latent guides.")

    if denoise < 1.0:
        keep = max(2, int(round((len(sigmas) - 1) * denoise)) + 1)
        sigmas = sigmas[-keep:]

    return sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas)
