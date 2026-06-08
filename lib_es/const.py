# Adaptive Progressive
AP_EULER_A_END = "exs_ap_euler_a_end"
AP_DPM_2M_END = "exs_ap_dpm_2m_end"
AP_ANCESTRAL_ETA = "exs_ap_ancestral_eta"
AP_DETAIL_STRENGTH = "exs_ap_detail_strength"

# Langevin Euler
LANGEVIN_STRENGTH = "exs_langevin_strength"

# Extended Reverse-Time
ER_MAX_STAGE = "er_max_stage"

# Gradient Estimation
GE_GAMMA = "ge_gamma"
GE_GAMMA_OFFSET = "ge_gamma_offset"
GE_USE_ADAPTIVE_STEPS = "ge_use_adaptive_steps"
GE_USE_TIMESTEP_ADAPTIVE_GAMMA = "ge_use_timestep_adaptive_gamma"
GE_VALIDATE_SCHEDULE = "ge_validate_schedule"

GE_DEFAULT_GAMMA = 2.0
GE_MIN_GAMMA = 1.0
GE_MAX_GAMMA = 3.0
GE_DEFAULT_GAMMA_OFFSET = 0.0
GE_MIN_GAMMA_OFFSET = -1.0
GE_MAX_GAMMA_OFFSET = 1.0

# DEIS
DEIS_MAX_ORDER = "deis_max_order"
DEIS_MODE = "deis_mode"
DEIS_DEFAULT_MAX_ORDER = 3
DEIS_DEFAULT_MODE = "tab"

# SEEDS-2
SEEDS_2_SOLVER_TYPE = "seeds_2_solver_type"
SEEDS_2_R = "seeds_2_r"
SEEDS_2_DEFAULT_SOLVER_TYPE = "phi_1"
SEEDS_2_DEFAULT_R = 0.5

# SEEDS-3
SEEDS_3_R1 = "seeds_3_r1"
SEEDS_3_R2 = "seeds_3_r2"
SEEDS_3_DEFAULT_R1 = 1.0 / 3.0
SEEDS_3_DEFAULT_R2 = 2.0 / 3.0

# SA-Solver
SA_SOLVER_PREDICTOR_ORDER = "sa_solver_predictor_order"
SA_SOLVER_CORRECTOR_ORDER = "sa_solver_corrector_order"
SA_SOLVER_USE_PECE = "sa_solver_use_pece"
SA_SOLVER_SIMPLE_ORDER_2 = "sa_solver_simple_order_2"
SA_SOLVER_DEFAULT_PREDICTOR_ORDER = 3
SA_SOLVER_DEFAULT_CORRECTOR_ORDER = 4
SA_SOLVER_START_PERCENT = 0.2
SA_SOLVER_END_PERCENT = 0.8

# ER-SDE VP variant
ER_SDE_VP_MAX_STAGE = "er_sde_vp_max_stage"
ER_SDE_VP_INTEGRATION_POINTS = 200
