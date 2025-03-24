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
        "exs_ld_cfg_alpha",
        OptionInfo(
            0.3,
            "CFG Alpha",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section,
        ).info("Controls the maximum increase in CFG scale that can occur during sampling."),
    )
    opts.add_option(
        "exs_ld_cfg_beta",
        OptionInfo(
            2.0,
            "CFG Beta",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 5.0, "step": 0.01},
            section=section,
        ).info("Controls how quickly CFG changes occur and where in the diffusion process they're concentrated."),
    )
    opts.add_option(
        "exs_ld_cfg_gamma",
        OptionInfo(
            0.1,
            "CFG Gamma",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section,
        ).info("Controls reduction of CFG at very low noise levels (end of sampling)."),
    )


on_ui_settings(on_settings)
