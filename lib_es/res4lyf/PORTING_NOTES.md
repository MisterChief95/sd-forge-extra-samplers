# RES4LYF Forge Port Notes

This package contains the Forge-facing RES4LYF sampler port used by
`sd-forge-extra-samplers`. The source of truth for the upstream code is the
local clone at `extensions/RES4LYF`, currently matching:

- Project: RES4LYF
- Upstream: https://github.com/ClownsharkBatwing/RES4LYF
- Commit: `119679d8d8d26e6db52757e705488abb6399d7d4`
- Commit date: 2026-05-20
- Primary author from git history: ClownsharkBatwing
- First upstream commit date: 2024-06-10

The upstream `LICENSE` file is AGPLv3 with an additional project notice at the
top prohibiting commercial service use without permission or a separate
commercial license from the copyright holder. The copied portions in this
package retain that AGPLv3 attribution in each Python file header.

## Comparison Summary

Compared against `extensions/RES4LYF` before local attribution headers were
added:

- Identical upstream code: `helper.py`, `latents.py`,
  `beta/constants.py`, `beta/deis_coefficients.py`, `beta/phi_functions.py`,
  `beta/rk_method_beta.py`, `beta/rk_noise_sampler_beta.py`.
- Small Forge compatibility edits: `beta/noise_classes.py`,
  `beta/rk_coefficients_beta.py`, `beta/rk_guide_func_beta.py`,
  `beta/rk_sampler_beta.py`.
- Forge shim or reduced modules: `__init__.py`, `beta/__init__.py`,
  `models.py`, `res4lyf.py`, `sigmas.py`, `style_transfer.py`.

## Intentional Divergences

- `__init__.py`: replaces upstream ComfyUI node registration and scheduler
  registration with a small compatibility shim that provides the `comfy.*`
  module symbols needed by the retained sampler code inside Forge.
- `beta/__init__.py`: imports only `rk_sampler_beta`; upstream ComfyUI node
  registration and beta node mappings are omitted because Forge exposes the
  samplers through `lib_es/extra_samplers/res4lyf_beta.py`.
- `models.py`: keeps only the `PRED` model-sampling type helper required by
  the sampler path; upstream model patcher and loader nodes are omitted because
  Forge already owns model loading and patching.
- `res4lyf.py`: keeps `RESplain()` logging and
  `get_display_sampler_category()` stubs; upstream initialization/UI behavior
  is not needed outside ComfyUI.
- `sigmas.py`: keeps a minimal `get_sigmas()` helper that delegates to the
  active Forge schedule/model instead of registering RES4LYF scheduler nodes.
- `style_transfer.py`: keeps only the spatial AdaIN/scattersort helpers used by
  `rk_sampler_beta.py`; upstream style-transfer nodes and model integrations
  are omitted.
- `beta/noise_classes.py`: makes `pywt` optional so Forge can import RES4LYF
  samplers without requiring wavelet noise dependencies unless that path is
  actually used.
- `beta/rk_coefficients_beta.py`: removes the unused `einops` import to avoid
  an unnecessary dependency in the Forge sampler path.
- `beta/rk_guide_func_beta.py`: provides a small fallback for the two `einops`
  `rearrange()` patterns used by the retained guide code when `einops` is not
  installed.
- `beta/rk_sampler_beta.py`: falls back from `tqdm.auto.trange` to `range` when
  `tqdm` is unavailable, and initializes missing Forge `model_options` /
  `transformer_options` dictionaries before writing regional-conditioning
  defaults.

## Wrapper Code

`lib_es/extra_samplers/res4lyf_beta.py` is Forge integration code, not a direct
copy of an upstream RES4LYF file. It adapts Forge denoiser/model objects to the
subset of ComfyUI-style interfaces expected by the retained RES4LYF sampler
implementation.
