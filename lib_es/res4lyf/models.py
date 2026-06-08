#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/models.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
from comfy.model_sampling import CONST, EDM, EPS, IMG_TO_IMG, V_PREDICTION, X0


class PRED:
    TYPE_VP = {CONST}
    TYPE_VE = {EPS}
    TYPE_VPRED = {V_PREDICTION, EDM}
    TYPE_X0 = {X0, IMG_TO_IMG}
    TYPE_ALL = TYPE_VP | TYPE_VE | TYPE_VPRED | TYPE_X0

    @classmethod
    def get_type(cls, model_sampling):
        bases = type(model_sampling).__mro__
        return next((v_type for v_type in bases if v_type in cls.TYPE_ALL), None)
