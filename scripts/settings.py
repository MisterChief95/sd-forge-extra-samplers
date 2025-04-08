from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts
import gradio as gr

import lib_es.const as consts


section = ("exs", "Extra Samplers")


def on_settings():
    opts.add_option(
        consts.AP_EULER_A_END,
        OptionInfo(
            0.35,
            "Euler A End",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.05},
            section=section,
        ),
    )
    opts.add_option(
        consts.AP_DPM_2M_END,
        OptionInfo(
            0.75,
            "DPM++ 2M End",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.05},
            section=section,
        ),
    )
    opts.add_option(
        consts.AP_ANCESTRAL_ETA,
        OptionInfo(
            0.4,
            "Adaptive Progressive Eta",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section,
        ),
    )
    opts.add_option(
        consts.AP_DETAIL_STRENGTH,
        OptionInfo(
            1.5,
            "Adaptive Progressive Detail Strength",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 3.0, "step": 0.01},
            section=section,
        ),
    )

    opts.add_option(
        consts.LANGEVIN_STRENGTH,
        OptionInfo(
            0.1,
            "Langevin Strength",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section,
        ),
    )

    opts.add_option(
        consts.ER_MAX_STAGE,
        OptionInfo(
            3,
            "Extended Reverse Time Max Stage",
            component=gr.Slider,
            component_args={"minimum": 1, "maximum": 3, "step": 1},
            section=section,
        ),
    )

    opts.add_option(
        consts.GE_GAMMA,
        OptionInfo(
            2.0,
            "Gradient Estimation Gamma",
            component=gr.Slider,
            component_args={"minimum": 1.0, "maximum": 5.0, "step": 0.1},
            section=section,
        ),
    )

    opts.add_option(
        consts.GE_GAMMA_OFFSET,
        OptionInfo(
            0.0,
            "Gradient Estimation Gamma Offset",
            component=gr.Slider,
            component_args={"minimum": -5.0, "maximum": 5.0, "step": 0.1},
            section=section,
        ),
    )

    opts.add_option(
        consts.GE_USE_ADAPTIVE_STEPS,
        OptionInfo(
            True,
            "Use Adaptive Steps",
            component=gr.Checkbox,
            section=section,
        ),
    )

    opts.add_option(
        consts.GE_USE_TIMESTEP_ADAPTIVE_GAMMA,
        OptionInfo(
            False,
            "Use Timestep Adaptive Gamma",
            component=gr.Checkbox,
            section=section,
        ),
    )


on_ui_settings(on_settings)
