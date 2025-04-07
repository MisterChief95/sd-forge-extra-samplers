import lib_es.extra_samplers

from modules.sd_samplers import add_sampler, all_samplers
from modules.sd_samplers_common import SamplerData
from modules.sd_samplers_kdiffusion import KDiffusionSampler


# See modules_forge/alter_samplers.py for the basis of this class and build_constructor function
class ExtraSampler(KDiffusionSampler):
    """
    Overloads KDiffusionSampler to add extra parameters to the constructor
    Based off lllyasviel's AlterSampler
    """

    def __init__(self, sd_model, sampler_name, options=None):
        self.sampler_name = sampler_name
        self.unet = sd_model.forge_objects.unet
        sampler_function = getattr(lib_es.extra_samplers, sampler_name)

        super().__init__(sampler_function, sd_model, options)

        self.extra_params = ["s_churn", "s_tmin", "s_tmax", "s_noise"]


def build_constructor(sampler_name):
    def constructor(m):
        return ExtraSampler(m, sampler_name)

    return constructor


euler_sampler_list = [
    (
        "Adaptive Progressive",
        "sample_adaptive_progressive",
        ["k_adaptive_progress_sampler"],
        {"scheduler": "sgm_uniform", "uses_ensd": True},
    ),
    ("Euler Max", "sample_euler_max", ["k_euler_max"], {}),
    ("Euler Negative", "sample_euler_negative", ["k_euler_negative"], {}),
    ("Euler Dy", "sample_euler_dy", ["k_euler_dy"], {}),
    ("Euler Dy Negative", "sample_euler_dy_negative", ["k_euler_dy_negative"], {}),
    ("Euler SMEA", "sample_euler_smea", ["k_euler_smea"], {}),
    ("Euler SMEA Dy", "sample_euler_smea_dy", ["k_euler_smea_dy"], {}),
    ("Euler SMEA Dy Negative", "sample_euler_smea_dy_negative", ["k_euler_smea_dy_negative"], {}),
    ("Extended Reverse SDE", "sample_er_sde", ["k_er_sde"], {"uses_ensd": True}),
    ("Gradient Estimation", "sample_gradient_estimation", ["k_gradient_estimation"], {}),
    ("Heun Ancestral", "sample_heun_ancestral", ["k_heun_ancestral"], {"uses_ensd": True}),
    ("Kohaku LoNyu Yog", "sample_kohaku_lonyu_yog", ["k_kohaku_lonyu_yog"], {}),
    (
        "Langevin Euler",
        "sample_langevin_euler",
        ["k_langevin_euler"],
        {"scheduler": "sgm_uniform"},
    ),
    ("Res Multistep", "sample_res_multistep", ["k_res_multistep"], {}),
    ("Res Multistep CFG++", "sample_res_multistep_cfg_pp", ["k_res_multistep_cfg_pp"], {}),
    ("Res Multistep Ancestral", "sample_res_multistep_ancestral", ["k_res_multistep_ancestral"], {"uses_ensd": True},),
    (
        "Res Multistep Ancestral CFG++",
        "sample_res_multistep_ancestral_cfg_pp",
        ["k_res_multistep_ancestral_cfg_pp"],
        {"uses_ensd": True},
    ),
]

samplers_data_k_diffusion: list[SamplerData] = [
    SamplerData(name, build_constructor(sampler_name=funcname), aliases, options)
    for name, funcname, aliases, options in euler_sampler_list
]

for sampler in samplers_data_k_diffusion:
    if sampler.name not in [x.name for x in all_samplers]:
        add_sampler(sampler)
