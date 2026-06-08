#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/sigmas.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
import torch


def get_sigmas(model, scheduler, steps, denoise, shift=0.0, lq_inflection_percent=0.5, lq_threshold_noise=0.025):
    if hasattr(model, "inner_model") and hasattr(model.inner_model, "get_sigmas"):
        sigmas = model.inner_model.get_sigmas(steps)
    elif hasattr(model, "get_sigmas"):
        sigmas = model.get_sigmas(steps)
    else:
        raise NotImplementedError("RES4LYF Forge port could not resolve a sigma scheduler for latent guides.")

    if denoise < 1.0:
        keep = max(2, int(round((len(sigmas) - 1) * denoise)) + 1)
        sigmas = sigmas[-keep:]

    return sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas)
