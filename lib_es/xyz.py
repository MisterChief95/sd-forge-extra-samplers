from modules import scripts

import lib_es.const as consts


def _grid_reference():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in (
            "scripts.xyz_grid",
            "xyz_grid.py",
        ) and hasattr(data, "module"):
            return data.module

    raise SystemError("Could not find X/Y/Z Plot...")


def xyz_support(cache: dict):
    def apply_field(field):
        def _(p, x, xs):
            cache.update({field: x})

        return _

    xyz_grid = _grid_reference()

    extra_axis_options = [
        xyz_grid.AxisOption("[Adaptive Prog] Euler A End", float, apply_field(consts.AP_EULER_A_END)),
        xyz_grid.AxisOption("[Adaptive Prog] DPM++ 2M End", float, apply_field(consts.AP_DPM_2M_END)),
        xyz_grid.AxisOption("[Adaptive Prog] Ancestral Eta", float, apply_field(consts.AP_ANCESTRAL_ETA)),
        xyz_grid.AxisOption("[Adaptive Prog] Detail Strength", float, apply_field(consts.AP_DETAIL_STRENGTH)),
        xyz_grid.AxisOption("[Langevin Euler] Langevin Strength", float, apply_field(consts.LANGEVIN_STRENGTH)),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)
