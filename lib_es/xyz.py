from modules import scripts


xyz_cache = {}


def _grid_reference():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in (
            "scripts.xyz_grid",
            "xyz_grid.py",
        ) and hasattr(data, "module"):
            return data.module

    raise SystemError("Could not find X/Y/Z Plot...")


def apply_field(field):
    def _(p, x, xs):
        xyz_cache.update({field: x})

    return _


xyz_grid = _grid_reference()

extra_axis_options = [
    xyz_grid.AxisOption("[Adaptive Prog] Detail Strength", float, apply_field("ap_detail_strength")),
    xyz_grid.AxisOption("[Adaptive Prog] Eta", float, apply_field("ap_eta")),
    xyz_grid.AxisOption("[Langevin dyn] CFG Alpha", float, apply_field("cfg_alpha")),
    xyz_grid.AxisOption("[Langevin dyn] CFG Beta", float, apply_field("cfg_beta")),
    xyz_grid.AxisOption("[Langevin dyn] CFG Gamma", float, apply_field("cfg_gamma")),
    xyz_grid.AxisOption("[Langevin dyn] Langevin Strength", float, apply_field("langevin_strength")),
]

xyz_grid.axis_options.extend(extra_axis_options)
