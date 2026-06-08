#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/res4lyf.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
def RESplain(*args, debug="info"):
    if debug == "debug" or debug is True:
        return
    if args:
        print("(RES4LYF Forge port)", " ".join(map(str, args)))


def get_display_sampler_category():
    return False
