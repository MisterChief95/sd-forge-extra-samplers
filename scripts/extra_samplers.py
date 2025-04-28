from typing import Any

import gradio as gr

import modules.scripts as scripts
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts
from modules.shared import opts

import lib_es.const as consts
from lib_es.xyz import xyz_support


def from_setting_or_default(key: str, default: None | Any) -> None | Any:
    return opts.data.get(key, default)


def on_change_update_setting(key: str, value: Any) -> None:
    opts.set(key, value)


class ExtraSamplerExtension(scripts.Script):
    def __init__(self):
        super().__init__()
        self.xyz_cache = {}
        xyz_support(self.xyz_cache)

    def title(self):
        return "Extra Samplers"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label="Extra Samplers", open=False):
            with gr.Accordion(label="Adaptive Progressive", open=False):
                gr.Markdown("Adaptive progressive sampler that automatically adjusts to different step counts. ")
                gr.Markdown(
                    "Phase ends are automatically adjusted based on the total number of steps. These are approximations"
                )
                with gr.Row():
                    euler_a_end = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=from_setting_or_default(consts.AP_EULER_A_END, 0.35),
                        label="Euler A end",
                    )
                    dpm_2m_end = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=from_setting_or_default(consts.AP_DPM_2M_END, 0.75),
                        label="DPM++ 2M end",
                    )
                with gr.Row():
                    ancestral_eta = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=from_setting_or_default(consts.AP_ANCESTRAL_ETA, 0.4),
                        label="Ancestral Eta",
                    )
                    detail_strength = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=from_setting_or_default(consts.AP_DETAIL_STRENGTH, 1.5),
                        label="Detail Strength",
                    )

                euler_a_end.change(
                    fn=lambda value: on_change_update_setting(consts.AP_EULER_A_END, value), inputs=[euler_a_end]
                )
                dpm_2m_end.change(
                    fn=lambda value: on_change_update_setting(consts.AP_DPM_2M_END, value), inputs=[dpm_2m_end]
                )
                ancestral_eta.change(
                    fn=lambda value: on_change_update_setting(consts.AP_ANCESTRAL_ETA, value),
                    inputs=[ancestral_eta],
                )
                detail_strength.change(
                    fn=lambda value: on_change_update_setting(consts.AP_DETAIL_STRENGTH, value),
                    inputs=[detail_strength],
                )

            with gr.Accordion(label="Langevin Euler", open=False):
                langevin_strength = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    step=0.01,
                    value=from_setting_or_default(consts.LANGEVIN_STRENGTH, 0.1),
                    label="Langevin Strength",
                    info="Langevin strength for Langevin Euler sampler. Adjust to control the amount of noise.",
                )
                langevin_strength.change(
                    fn=lambda value: on_change_update_setting(consts.LANGEVIN_STRENGTH, value),
                    inputs=[langevin_strength],
                )

            with gr.Accordion(label="Gradient Estimation", open=False):
                use_adaptive_steps = from_setting_or_default(consts.GE_USE_ADAPTIVE_STEPS, False)

                adaptive_steps = gr.Checkbox(
                    label="Use Adaptive Steps",
                    value=use_adaptive_steps,
                    info="Modify the number of steps based on the noise schedule.",
                )
                use_timestep_adaptive_gamma = gr.Checkbox(
                    label="Timestep-Based Adaptive Gamma",
                    value=from_setting_or_default(consts.GE_USE_TIMESTEP_ADAPTIVE_GAMMA, False),
                    info="Adjust gamma during generation.",
                )
                gamma = gr.Slider(
                    minimum=consts.GE_MIN_GAMMA,
                    maximum=consts.GE_MAX_GAMMA,
                    step=0.05,
                    value=from_setting_or_default(consts.GE_GAMMA, consts.GE_DEFAULT_GAMMA),
                    label="Gamma",
                    info="Gamma value for gradient estimation. Higher values increase the amount of noise.",
                    interactive=not use_adaptive_steps,
                )
                gamma_offset = gr.Slider(
                    minimum=consts.GE_MIN_GAMMA_OFFSET,
                    maximum=consts.GE_MAX_GAMMA_OFFSET,
                    step=0.05,
                    value=from_setting_or_default(consts.GE_GAMMA_OFFSET, consts.GE_DEFAULT_GAMMA_OFFSET),
                    label="Gamma Offset",
                    info="Offset to add to the calculated gamma when using adaptive steps.",
                    interactive=use_adaptive_steps,
                )
                gamma.change(fn=lambda value: on_change_update_setting(consts.GE_GAMMA, value), inputs=[gamma])
                gamma_offset.change(
                    fn=lambda value: on_change_update_setting(consts.GE_GAMMA_OFFSET, value), inputs=[gamma_offset]
                )

                # Update interactivity when adaptive steps checkbox changes
                adaptive_steps.change(
                    fn=lambda value: (gr.Slider(interactive=not value), gr.Slider(interactive=value)),
                    inputs=[adaptive_steps],
                    outputs=[gamma, gamma_offset],
                    js=None,
                ).then(
                    fn=lambda value: on_change_update_setting(consts.GE_USE_ADAPTIVE_STEPS, value),
                    inputs=[adaptive_steps],
                )

                use_timestep_adaptive_gamma.change(
                    fn=lambda value: on_change_update_setting(consts.GE_USE_TIMESTEP_ADAPTIVE_GAMMA, value),
                    inputs=[use_timestep_adaptive_gamma],
                )

                validate_schedule = gr.Checkbox(
                    label="Validate Schedule",
                    value=from_setting_or_default(consts.GE_VALIDATE_SCHEDULE, False),
                    info="Validate the noise schedule (For debugging purposes).",
                )

            with gr.Accordion(label="Extended Reverse SDE", open=False):
                gr.Markdown("Extended reverse SDE sampler.")
                gr.Markdown("Max stage for extended reverse SDE.")
                max_stage = gr.Slider(
                    minimum=1,
                    maximum=3,
                    step=1,
                    value=from_setting_or_default(consts.ER_MAX_STAGE, 3),
                    label="Max Stage",
                )
                max_stage.change(fn=lambda value: on_change_update_setting(consts.MAX_STAGE, value), inputs=[max_stage])

        return [
            euler_a_end,
            dpm_2m_end,
            ancestral_eta,
            detail_strength,
            langevin_strength,
            max_stage,
            adaptive_steps,
            use_timestep_adaptive_gamma,
            gamma,
            gamma_offset,
            validate_schedule,
        ]

    def get_values_and_apply(self, p: StableDiffusionProcessing, values: dict):
        for key, value in values.items():
            value = self.xyz_cache.pop(key, value)
            setattr(p, key, value)
            p.extra_generation_params[key] = value

    def process_batch(
        self,
        p: StableDiffusionProcessing,
        euler_a_end: float,
        dpm_2m_end: float,
        ancestral_eta: float,
        detail_strength: float,
        langevin_strength: float,
        max_stage: int,
        use_adaptive_steps: bool,
        use_timestep_adaptive_gamma: bool,
        gamma: float,
        gamma_offset: float,
        validate_schedule: bool,
        batch_number: int,
        prompts: list[str],
        seeds: list[int],
        subseeds: list[int],
    ):
        if p.sampler_name == "Adaptive Progressive":
            self.get_values_and_apply(
                p,
                {
                    consts.AP_EULER_A_END: euler_a_end,
                    consts.AP_DPM_2M_END: dpm_2m_end,
                    consts.AP_ANCESTRAL_ETA: ancestral_eta,
                    consts.AP_DETAIL_STRENGTH: detail_strength,
                },
            )
        elif p.sampler_name == "Langevin Euler":
            self.get_values_and_apply(p, {consts.LANGEVIN_STRENGTH: langevin_strength})
        elif p.sampler_name == "Gradient Estimation":
            self.get_values_and_apply(
                p,
                {
                    consts.GE_GAMMA: gamma,
                    consts.GE_GAMMA_OFFSET: gamma_offset,
                    consts.GE_USE_ADAPTIVE_STEPS: use_adaptive_steps,
                    consts.GE_USE_TIMESTEP_ADAPTIVE_GAMMA: use_timestep_adaptive_gamma,
                    consts.GE_VALIDATE_SCHEDULE: validate_schedule,
                },
            )
        elif p.sampler_name == "Extended Reverse SDE":
            self.get_values_and_apply(p, {consts.ER_MAX_STAGE: max_stage})


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
            consts.GE_DEFAULT_GAMMA,
            "Gradient Estimation Gamma",
            component=gr.Slider,
            component_args={"minimum": consts.GE_MIN_GAMMA, "maximum": consts.GE_MAX_GAMMA, "step": 0.1},
            section=section,
        ),
    )

    opts.add_option(
        consts.GE_GAMMA_OFFSET,
        OptionInfo(
            consts.GE_DEFAULT_GAMMA_OFFSET,
            "Gradient Estimation Gamma Offset",
            component=gr.Slider,
            component_args={"minimum": consts.GE_MIN_GAMMA_OFFSET, "maximum": consts.GE_MAX_GAMMA_OFFSET, "step": 0.1},
            section=section,
        ),
    )

    opts.add_option(
        consts.GE_USE_ADAPTIVE_STEPS,
        OptionInfo(
            False,
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