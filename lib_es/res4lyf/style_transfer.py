#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/style_transfer.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
import torch


def _flatten_spatial(x):
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])


def _unflatten_spatial(x, shape):
    return x.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)


def apply_scattersort_spatial(x_spatial: torch.Tensor, y_spatial: torch.Tensor):
    x_emb = _flatten_spatial(x_spatial)
    y_emb = _flatten_spatial(y_spatial)

    x_sorted, x_idx = x_emb.sort(dim=-2)
    y_sorted, _ = y_emb.sort(dim=-2)
    x_emb = x_sorted.scatter(dim=-2, index=x_idx, src=y_sorted.expand(x_sorted.shape))

    return _unflatten_spatial(x_emb, x_spatial.shape)


def apply_adain_spatial(x_spatial: torch.Tensor, y_spatial: torch.Tensor):
    x_emb = _flatten_spatial(x_spatial)
    y_emb = _flatten_spatial(y_spatial)

    x_mean = x_emb.mean(-2, keepdim=True)
    x_std = x_emb.std(-2, keepdim=True).clamp_min(1e-8)
    y_mean = y_emb.mean(-2, keepdim=True)
    y_std = y_emb.std(-2, keepdim=True).clamp_min(1e-8)

    x_emb_adain = (x_emb - x_mean) / x_std
    x_emb_adain = (x_emb_adain * y_std) + y_mean

    return _unflatten_spatial(x_emb_adain, x_spatial.shape)
