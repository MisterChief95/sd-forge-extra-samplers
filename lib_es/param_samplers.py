from .xyz import xyz_cache

from modules.shared import opts


class XYZSamplerSupport:
    def cache_pop(self, key, default=None):
        return xyz_cache.pop(key, default)


class DynamicCFGSampler(XYZSamplerSupport):
    def get_cfg_values(self):
        return (
            super().cache_pop("cfg_alpha", opts.data.get("exs_cfg_alpha", 0.3)),
            super().cache_pop("cfg_beta", opts.data.get("exs_cfg_beta", 2.0)),
            super().cache_pop("cfg_gamma", opts.data.get("exs_cfg_gamma", 0.1)),
        )


class LangevinSampler(XYZSamplerSupport):
    def get_langevin_strength(self):
        return super().cache_pop("langevin_strength", opts.data.get("exs_langevin_strength", 0.1))


class AdaptiveProgressiveSampling(XYZSamplerSupport):
    def get_eta(self):
        return super().cache_pop("ap_eta", opts.data.get("exs_adaptive_prog_sampler_eta", 0.4))

    def get_detail_strength(self):
        return super().cache_pop("ap_detail_strength", opts.data.get("exs_adaptive_prog_sampler_detail_strength", 1.5))
