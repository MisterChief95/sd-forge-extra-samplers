import torch

from lib_es.utils import sampler_metadata


class _ForgeInnerModel:
    def __init__(self, sd_model, model_sampling, device):
        self._sd_model = sd_model
        self.model_sampling = model_sampling
        self.device = device
        self.diffusion_model = sd_model.forge_objects.unet.model.diffusion_model

    def __getattr__(self, name):
        return getattr(self._sd_model, name)

    def process_latent_in(self, samples):
        return samples


class _ForgeSchedule:
    def __init__(self, original_schedule, inner_model):
        self._original_schedule = original_schedule
        self.inner_model = inner_model
        self.predictor = inner_model.model_sampling
        self.conds = {}
        self.cfg = 1.0

    def __getattr__(self, name):
        return getattr(self._original_schedule, name)


class _ForgeDenoiser:
    def __init__(self, model, x):
        self._model = model
        original_schedule = model.inner_model
        sd_model = original_schedule.inner_model
        model_sampling = sd_model.forge_objects.unet.model.predictor
        inner_model = _ForgeInnerModel(sd_model, model_sampling, x.device)
        self.inner_model = _ForgeSchedule(original_schedule, inner_model)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def _sample_res4lyf(model, x, sigmas, extra_args, callback, disable, rk_type, eta=None, eta_substep=None):
    from lib_es.res4lyf.beta import rk_sampler_beta

    kwargs = {"rk_type": rk_type}
    if eta is not None:
        kwargs["eta"] = eta
    if eta_substep is not None:
        kwargs["eta_substep"] = eta_substep

    return rk_sampler_beta.sample_rk_beta(
        _ForgeDenoiser(model, x),
        x,
        sigmas,
        None,
        extra_args,
        callback,
        disable,
        **kwargs,
    ).to(dtype=x.dtype, device=x.device)


def _make_res_sampler(rk_type, ode=False):
    def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
        eta = 0.0 if ode else None
        eta_substep = 0.0 if ode else None
        return _sample_res4lyf(model, x, sigmas, extra_args, callback, disable, rk_type, eta, eta_substep)

    sampler.__name__ = f"sample_res4lyf_{rk_type}{'_ode' if ode else ''}"
    return torch.no_grad()(sampler)


sample_res4lyf_res_2m = sampler_metadata(
    "RES4LYF res_2m",
    {"scheduler": "sgm_uniform"},
    ["res_2m", "res4lyf_res_2m"],
)(_make_res_sampler("res_2m"))
sample_res4lyf_res_3m = sampler_metadata(
    "RES4LYF res_3m",
    {"scheduler": "sgm_uniform"},
    ["res_3m", "res4lyf_res_3m"],
)(_make_res_sampler("res_3m"))
sample_res4lyf_res_2s = sampler_metadata(
    "RES4LYF res_2s",
    {"scheduler": "sgm_uniform"},
    ["res_2s", "res4lyf_res_2s"],
)(_make_res_sampler("res_2s"))
sample_res4lyf_res_3s = sampler_metadata(
    "RES4LYF res_3s",
    {"scheduler": "sgm_uniform"},
    ["res_3s", "res4lyf_res_3s"],
)(_make_res_sampler("res_3s"))
sample_res4lyf_res_5s = sampler_metadata(
    "RES4LYF res_5s",
    {"scheduler": "sgm_uniform"},
    ["res_5s", "res4lyf_res_5s"],
)(_make_res_sampler("res_5s"))
sample_res4lyf_res_6s = sampler_metadata(
    "RES4LYF res_6s",
    {"scheduler": "sgm_uniform"},
    ["res_6s", "res4lyf_res_6s"],
)(_make_res_sampler("res_6s"))

sample_res4lyf_res_2m_ode = sampler_metadata(
    "RES4LYF res_2m ODE",
    {"scheduler": "sgm_uniform"},
    ["res_2m_ode", "res4lyf_res_2m_ode"],
)(_make_res_sampler("res_2m", ode=True))
sample_res4lyf_res_3m_ode = sampler_metadata(
    "RES4LYF res_3m ODE",
    {"scheduler": "sgm_uniform"},
    ["res_3m_ode", "res4lyf_res_3m_ode"],
)(_make_res_sampler("res_3m", ode=True))
sample_res4lyf_res_2s_ode = sampler_metadata(
    "RES4LYF res_2s ODE",
    {"scheduler": "sgm_uniform"},
    ["res_2s_ode", "res4lyf_res_2s_ode"],
)(_make_res_sampler("res_2s", ode=True))
sample_res4lyf_res_3s_ode = sampler_metadata(
    "RES4LYF res_3s ODE",
    {"scheduler": "sgm_uniform"},
    ["res_3s_ode", "res4lyf_res_3s_ode"],
)(_make_res_sampler("res_3s", ode=True))
sample_res4lyf_res_5s_ode = sampler_metadata(
    "RES4LYF res_5s ODE",
    {"scheduler": "sgm_uniform"},
    ["res_5s_ode", "res4lyf_res_5s_ode"],
)(_make_res_sampler("res_5s", ode=True))
sample_res4lyf_res_6s_ode = sampler_metadata(
    "RES4LYF res_6s ODE",
    {"scheduler": "sgm_uniform"},
    ["res_6s_ode", "res4lyf_res_6s_ode"],
)(_make_res_sampler("res_6s", ode=True))
