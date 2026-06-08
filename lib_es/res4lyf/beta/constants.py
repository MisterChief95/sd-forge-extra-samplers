#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/beta/constants.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
MAX_STEPS = 10000


IMPLICIT_TYPE_NAMES = [
    "rebound",
    "retro-eta",
    "bongmath",
    "predictor-corrector",
]


GUIDE_MODE_NAMES_BETA_SIMPLE = [
    "flow",
    "sync",
    "lure",
    "data",
    "epsilon",
    "inversion",
    "pseudoimplicit",
    "fully_pseudoimplicit",
    "none",
]

FRAME_WEIGHTS_CONFIG_NAMES = ["frame_weights", "frame_weights_inv", "frame_targets"]

FRAME_WEIGHTS_DYNAMICS_NAMES = [
    "constant",
    "linear",
    "ease_out",
    "ease_in",
    "middle",
    "trough",
]


FRAME_WEIGHTS_SCHEDULE_NAMES = [
    "moderate_early",
    "moderate_late",
    "fast_early",
    "fast_late",
    "slow_early",
    "slow_late",
]


GUIDE_MODE_NAMES_PSEUDOIMPLICIT = [
    "pseudoimplicit",
    "pseudoimplicit_cw",
    "pseudoimplicit_projection",
    "pseudoimplicit_projection_cw",
    "fully_pseudoimplicit",
    "fully_pseudoimplicit_projection",
    "fully_pseudoimplicit_cw",
    "fully_pseudoimplicit_projection_cw",
]
