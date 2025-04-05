from typing import Any

import gradio as gr

import modules.scripts as scripts
from modules.shared import opts
from modules.processing import StableDiffusionProcessing


from lib_es.xyz import xyz_support
import lib_es.const as consts


def from_setting_or_default(key: str, default: None | Any) -> None | Any:
    return opts.data.get(key, default)


def on_change_update_setting(key: str, value: Any) -> None:
    opts.data[key] = value


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

            with gr.Row():
                gr.Markdown("Langevin Strength")
                langevin_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=from_setting_or_default(consts.LANGEVIN_STRENGTH, 0.1),
                    label="Langevin Strength",
                )
                langevin_strength.change(
                    fn=lambda value: on_change_update_setting(consts.LANGEVIN_STRENGTH, value),
                    inputs=[langevin_strength],
                )

        return [
            euler_a_end,
            dpm_2m_end,
            ancestral_eta,
            detail_strength,
            langevin_strength,
        ]

    def get_values_and_apply(self, p: StableDiffusionProcessing, values: dict):
        for key, value in values.items():
            value = self.xyz_cache.pop(key, value)
            setattr(p, key, value)
            # p.extra_generation_params[key] = value

    def process_batch(
        self,
        p: StableDiffusionProcessing,
        euler_a_end: float,
        dpm_2m_end: float,
        ancestral_eta: float,
        detail_strength: float,
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
            self.get_values_and_apply(p, {consts.LANGEVIN_STRENGTH: detail_strength})
