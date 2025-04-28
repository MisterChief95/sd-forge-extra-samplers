from modules.sd_samplers import add_sampler, all_samplers
from modules.sd_samplers_common import SamplerData
from modules.sd_samplers_kdiffusion import KDiffusionSampler

from lib_es.extra_samplers import __sampler_funcs__


# See modules_forge/alter_samplers.py for the basis of this class and build_constructor function
class ExtraSampler(KDiffusionSampler):
    """
    Overloads KDiffusionSampler to add extra parameters to the constructor
    Based off lllyasviel's AlterSampler
    """

    def __init__(self, sd_model, sampler_name, sampler_func, options=None):
        self.sampler_name = sampler_name
        self.unet = sd_model.forge_objects.unet
        sampler_function = sampler_func
        super().__init__(sampler_function, sd_model, options)
        self.extra_params = ["s_churn", "s_tmin", "s_tmax", "s_noise"]


def build_constructor(sampler_name, sampler_func):
    def constructor(m):
        return ExtraSampler(m, sampler_name, sampler_func)

    return constructor


extra_sampler_list = [
    (
        fn.sampler_name,
        fn,
        [fn.sampler_k_name],
        fn.sampler_extra_params,
    )
    for fn in __sampler_funcs__
]

samplers_data_k_diffusion: list[SamplerData] = [
    SamplerData(name, build_constructor(sampler_name=name, sampler_func=funcname), aliases, options)
    for name, funcname, aliases, options in extra_sampler_list
]


def add_extra_samplers():
    for sampler in samplers_data_k_diffusion:
        if sampler.name not in [x.name for x in all_samplers]:
            add_sampler(sampler)
