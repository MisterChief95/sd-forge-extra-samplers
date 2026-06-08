from lib_es.extra_samplers.adaptive_progressive import sample_adaptive_progressive
from lib_es.extra_samplers.euler_dy import sample_euler_dy
from lib_es.extra_samplers.euler_dy_negative import sample_euler_dy_negative
from lib_es.extra_samplers.euler_smea import sample_euler_smea
from lib_es.extra_samplers.euler_smea_dy import sample_euler_smea_dy
from lib_es.extra_samplers.euler_smea_dy_negative import sample_euler_smea_dy_negative
from lib_es.extra_samplers.euler_max import sample_euler_max
from lib_es.extra_samplers.euler_multipass import (
    sample_euler_multipass,
    sample_euler_multipass_cfg_pp,
    sample_euler_ancestral_multipass,
    sample_euler_ancestral_multipass_cfg_pp,
)
from lib_es.extra_samplers.euler_negative import sample_euler_negative
from lib_es.extra_samplers.deis_sampler import sample_deis
from lib_es.extra_samplers.gradient_estimation import (
    sample_gradient_estimation,
    sample_gradient_estimation_cfg_pp,
)
from lib_es.extra_samplers.heun_ancestral import sample_heun_ancestral
from lib_es.extra_samplers.sa_solver_sampler import sample_sa_solver, sample_sa_solver_stable
from lib_es.extra_samplers.seeds_2 import sample_seeds_2
from lib_es.extra_samplers.seeds_3 import sample_seeds_3
from lib_es.extra_samplers.langevin_euler import sample_langevin_euler
from lib_es.extra_samplers.res_multistep import (
    sample_res_multistep,
    sample_res_multistep_cfg_pp,
    sample_res_multistep_ancestral,
    sample_res_multistep_ancestral_cfg_pp,
)
from lib_es.extra_samplers.res4lyf_beta import (
    sample_res4lyf_res_2m,
    sample_res4lyf_res_3m,
    sample_res4lyf_res_2s,
    sample_res4lyf_res_3s,
    sample_res4lyf_res_5s,
    sample_res4lyf_res_6s,
    sample_res4lyf_res_2m_ode,
    sample_res4lyf_res_3m_ode,
    sample_res4lyf_res_2s_ode,
    sample_res4lyf_res_3s_ode,
    sample_res4lyf_res_5s_ode,
    sample_res4lyf_res_6s_ode,
)
from lib_es.extra_samplers.ssprk3 import sample_ssprk3

__sampler_funcs__ = [
    sample_adaptive_progressive,
    sample_deis,
    sample_euler_max,
    sample_euler_negative,
    sample_euler_dy,
    sample_euler_dy_negative,
    sample_euler_smea,
    sample_euler_smea_dy,
    sample_euler_smea_dy_negative,
    sample_euler_multipass,
    sample_euler_multipass_cfg_pp,
    sample_euler_ancestral_multipass,
    sample_euler_ancestral_multipass_cfg_pp,
    sample_gradient_estimation,
    sample_gradient_estimation_cfg_pp,
    sample_heun_ancestral,
    sample_langevin_euler,
    sample_res_multistep_ancestral_cfg_pp,
    sample_res_multistep_ancestral,
    sample_res_multistep_cfg_pp,
    sample_res_multistep,
    sample_res4lyf_res_2m,
    sample_res4lyf_res_3m,
    sample_res4lyf_res_2s,
    sample_res4lyf_res_3s,
    sample_res4lyf_res_5s,
    sample_res4lyf_res_6s,
    sample_res4lyf_res_2m_ode,
    sample_res4lyf_res_3m_ode,
    sample_res4lyf_res_2s_ode,
    sample_res4lyf_res_3s_ode,
    sample_res4lyf_res_5s_ode,
    sample_res4lyf_res_6s_ode,
    sample_sa_solver,
    sample_sa_solver_stable,
    sample_seeds_2,
    sample_seeds_3,
    sample_ssprk3,
]
