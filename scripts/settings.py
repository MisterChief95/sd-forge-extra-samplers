from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts
import gradio as gr

section = ("exs", "Extra Samplers")


def on_settings():
    opts.add_option(
        "exs_adaptive_prog_sampler_eta",
        OptionInfo(
            0.4,
            "Adaptive Progressive Eta",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section,
        ),
    )
    opts.add_option(
        "exs_adaptive_prog_sampler_detail_strength",
        OptionInfo(
            1.5,
            "Adaptive Progressive Detail Strength",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 3.0, "step": 0.01},
            section=section,
        ),
    )

    opts.add_option(
        "exs_cfg_alpha",
        OptionInfo(
            0.3,
            "CFG Alpha",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section,
        ).info("Controls the maximum increase in CFG scale that can occur during sampling."),
    )
    opts.add_option(
        "exs_cfg_beta",
        OptionInfo(
            2.0,
            "CFG Beta",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 5.0, "step": 0.01},
            section=section,
        ).info("Controls how quickly CFG changes occur and where in the diffusion process they're concentrated."),
    )
    opts.add_option(
        "exs_cfg_gamma",
        OptionInfo(
            0.1,
            "CFG Gamma",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.01},
            section=section,
        ).info("Controls reduction of CFG at very low noise levels (end of sampling)."),
    )


on_ui_settings(on_settings)
