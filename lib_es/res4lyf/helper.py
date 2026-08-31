#
# Portions of this file are derived from RES4LYF
# Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/119679d8d8d26e6db52757e705488abb6399d7d4/helper.py
# Copyright (C) 2024 ClownsharkBatwing and RES4LYF contributors
#
# This specific portion is licensed under the GNU Affero General Public License
# Version 3 (AGPLv3). The rest of this file remains under GNU GPLv3.
# See PORTING_NOTES.md for Forge-specific divergence notes.
#
import torch
from typing import List

import re
import functools
import copy

from .res4lyf import RESplain


# EXTRA_OPTIONS OPS


class ExtraOptions:
    def __init__(self, extra_options):
        self.extra_options = extra_options
        self.mute = False

    # debugMode 0: Follow self.mute only
    # debugMode 1: Print with debug flag if not muted
    # debugMode 2: Never print
    def __call__(self, option, default=None, ret_type=None, match_all_flags=False, debugMode=0):
        if isinstance(option, (tuple, list)):
            if match_all_flags:
                return all(self(single_option, default, ret_type) for single_option in option)
            else:
                return any(self(single_option, default, ret_type) for single_option in option)

        if default is None:  # get flag
            pattern = rf"^(?:{re.escape(option)}\s*$|{re.escape(option)}=)"
            return bool(re.search(pattern, self.extra_options, flags=re.MULTILINE))
        elif ret_type is None:
            ret_type = type(default)

            if ret_type.__module__ != "builtins":
                mod = __import__(default.__module__)
                ret_type = lambda v: getattr(mod, v, None)

        if ret_type == list:
            pattern = rf"^{re.escape(option)}\s*=\s*([a-zA-Z0-9_.,+-]+)\s*$"
            match = re.search(pattern, self.extra_options, flags=re.MULTILINE)

            if match:
                value = match.group(1)
                if not self.mute and debugMode != 2:
                    if debugMode == 1:
                        RESplain("Set extra_option: ", option, "=", value, debug=True)
                    else:
                        RESplain("Set extra_option: ", option, "=", value)
            else:
                value = default

            if type(value) == str:
                value = value.split(",")

                if type(default[0]) == type:
                    ret_type = default[0]
                else:
                    ret_type = type(default[0])

                value = [ret_type(value[_]) for _ in range(len(value))]

        else:
            pattern = rf"^{re.escape(option)}\s*=\s*([a-zA-Z0-9_.+-]+)\s*$"
            match = re.search(pattern, self.extra_options, flags=re.MULTILINE)
            if match:
                if ret_type == bool:
                    value_str = match.group(1).lower()
                    value = value_str in ("true", "1", "yes", "on")
                else:
                    value = ret_type(match.group(1))
                if not self.mute and debugMode != 2:
                    if debugMode == 1:
                        RESplain("Set extra_option: ", option, "=", value, debug=True)
                    else:
                        RESplain("Set extra_option: ", option, "=", value)
            else:
                value = default

        return value


def extra_options_flag(flag, extra_options):
    pattern = rf"^(?:{re.escape(flag)}\s*$|{re.escape(flag)}=)"
    return bool(re.search(pattern, extra_options, flags=re.MULTILINE))


def get_extra_options_kv(key, default, extra_options, ret_type=None):
    ret_type = type(default) if ret_type is None else ret_type

    pattern = rf"^{re.escape(key)}\s*=\s*([a-zA-Z0-9_.+-]+)\s*$"
    match = re.search(pattern, extra_options, flags=re.MULTILINE)

    if match:
        value = match.group(1)
    else:
        value = default

    return ret_type(value)




# MISCELLANEOUS OPS


def has_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True







# COMFY OPS


def is_video_model(model):
    is_video_model = False
    try:
        is_video_model = (
            "video" in model.inner_model.inner_model.model_config.unet_config["image_model"]
            or "cosmos" in model.inner_model.inner_model.model_config.unet_config["image_model"]
            or "wan2" in model.inner_model.inner_model.model_config.unet_config["image_model"]
            or "ltxv" in model.inner_model.inner_model.model_config.unet_config["image_model"]
        )
    except:
        pass
    return is_video_model




# MISC OPS


def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor



class PrecisionTool:
    def __init__(self, cast_type="fp64"):
        self.cast_type = cast_type

    def cast_tensor(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.cast_type not in ["fp64", "fp32", "fp16"]:
                return func(*args, **kwargs)

            target_device = None
            for arg in args:
                if torch.is_tensor(arg):
                    target_device = arg.device
                    break
            if target_device is None:
                for v in kwargs.values():
                    if torch.is_tensor(v):
                        target_device = v.device
                        break

            # recursively zs_recast tensors in nested dictionaries
            def cast_and_move_to_device(data):
                if torch.is_tensor(data):
                    if self.cast_type == "fp64":
                        return data.to(torch.float64).to(target_device)
                    elif self.cast_type == "fp32":
                        return data.to(torch.float32).to(target_device)
                    elif self.cast_type == "fp16":
                        return data.to(torch.float16).to(target_device)
                elif isinstance(data, dict):
                    return {k: cast_and_move_to_device(v) for k, v in data.items()}
                return data

            new_args = [cast_and_move_to_device(arg) for arg in args]
            new_kwargs = {k: cast_and_move_to_device(v) for k, v in kwargs.items()}

            return func(*new_args, **new_kwargs)

        return wrapper

    def set_cast_type(self, new_value):
        if new_value in ["fp64", "fp32", "fp16"]:
            self.cast_type = new_value
        else:
            self.cast_type = "fp64"


precision_tool = PrecisionTool(cast_type="fp64")


class FrameWeightsManager:
    def __init__(self):
        self._weight_configs = {}

        self._default_config = {
            "frame_weights": None,  # Tensor of weights if directly specified
            "dynamics": "linear",  # Function type for dynamic period
            "schedule": "moderate_early",  # Schedule type
            "scale": 0.5,  # Amount of change
            "is_reversed": False,  # Whether to reverse weights
            "custom_string": None,  # Per-configuration custom string
        }
        self.dtype = torch.float64
        self.device = torch.device("cpu")

    def set_device_and_dtype(self, device=None, dtype=None):
        """Set the device and dtype for generated weights"""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def set_custom_weights(self, config_name, weights):
        """Set custom weights for a specific configuration"""
        if config_name not in self._weight_configs:
            self._weight_configs[config_name] = self._default_config.copy()

        self._weight_configs[config_name]["frame_weights"] = weights
        return self

    def add_weight_config(self, name, **kwargs):
        if name not in self._weight_configs:
            self._weight_configs[name] = self._default_config.copy()

        for key, value in kwargs.items():
            if key in self._default_config:
                self._weight_configs[name][key] = value
            # ignore unknown parameters

        return self

    def get_weight_config(self, name):
        if name not in self._weight_configs:
            return None
        return self._weight_configs[name].copy()

    def get_frame_weights_by_name(self, name, num_frames, step=None):
        config = self.get_weight_config(name)
        if config is None:
            return None

        weights_tensor = self._generate_frame_weights(
            num_frames,
            config["dynamics"],
            config["schedule"],
            config["scale"],
            config["is_reversed"],
            config["frame_weights"],
            step=step,
            custom_string=config["custom_string"],
        )

        if config["custom_string"] is not None and config["custom_string"].strip() != "" and weights_tensor is not None:
            # ensure that the custom_string has more than just lines that begin with non-numeric characters
            custom_string = config["custom_string"].strip()
            custom_string = re.sub(r"^[^0-9].*", "", custom_string, flags=re.MULTILINE)
            custom_string = re.sub(r"^\s*$", "", custom_string, flags=re.MULTILINE)
            if custom_string.strip() != "":
                # If the custom_string is not empty, show the custom weights
                formatted_weights = [f"{w:.2f}" for w in weights_tensor.tolist()]
                RESplain(f"Custom '{name}' for step {step}: {formatted_weights}", debug=True)
        elif weights_tensor is None:
            weights_tensor = torch.ones(num_frames, dtype=self.dtype, device=self.device)

        return weights_tensor

    def _generate_custom_weights(self, num_frames, custom_string, step=None):
        """
        Generate custom weights based on the provided frame weights from a string with one line per step.

        Args:
            num_frames: Number of frames to generate weights for
            custom_string: The custom weights string to parse
            step: Specific step to use (0-indexed). If None, uses the last line.

        Features:
        - Each line represents weights for one step
        - Add *[multiplier] at the end of a line to scale those weights (e.g., "1.0, 0.8, 0.6*1.5")
        - Include "interpolate" on its own line to interpolate each line to match num_frames
        - Prefix line with the steps to apply it to (e.g. "0-5: 1.0, 0.8, 0.6")

        Example:
        0-5:1.0, 0.8, 0.6, 0.4, 0.2, 0.0
        6-10:0.0, 0.2, 0.4, 0.6, 0.8, 1.0*1.5
        11-30:0.0, 0.5, 1.0, 0.5, 0.0, 0.0*0.8
        interpolate
        """
        if custom_string is not None:
            interpolate_frames = "interpolate" in custom_string

            lines = custom_string.strip().split("\n")
            lines = [line for line in lines if line.strip() and not line.strip().startswith("interp")]

            if not lines:
                return None

            if step is not None:
                matching_line = None
                for line in lines:
                    # Check if line has a step range prefix
                    step_range_match = re.match(r"^(\d+)-(\d+):(.*)", line.strip())
                    if step_range_match:
                        start_step = int(step_range_match.group(1))
                        end_step = int(step_range_match.group(2))
                        if start_step <= step <= end_step:
                            matching_line = step_range_match.group(3).strip()

                if matching_line is not None:
                    weights_str = matching_line
                else:
                    # if no matching line, try to use the step number line or the last line
                    if step < len(lines):
                        line_index = step
                    else:
                        line_index = len(lines) - 1

                    if line_index < 0:
                        return None

                    weights_str = lines[line_index].strip()

                    if ":" in weights_str:
                        weights_str = weights_str.split(":", 1)[1].strip()
            else:
                # When no specific step is provided, use the last line
                line_index = len(lines) - 1
                weights_str = lines[line_index].strip()
                if ":" in weights_str:
                    weights_str = weights_str.split(":", 1)[1].strip()

            if not weights_str:
                return None

            multiplier = 1.0
            if "*" in weights_str:
                parts = weights_str.rsplit("*", 1)
                if len(parts) == 2:
                    weights_str = parts[0].strip()
                    try:
                        multiplier = float(parts[1].strip())
                    except ValueError:
                        RESplain(f"Invalid multiplier format: {parts[1]}")

            try:
                weights = [float(w.strip()) for w in weights_str.split(",")]
                weights_tensor = torch.tensor(weights, dtype=self.dtype, device=self.device)

                if multiplier != 1.0:
                    weights_tensor = weights_tensor * multiplier

                if interpolate_frames and len(weights_tensor) != num_frames:
                    if len(weights_tensor) > 1:
                        orig_positions = torch.linspace(0, 1, len(weights_tensor), dtype=self.dtype, device=self.device)
                        new_positions = torch.linspace(0, 1, num_frames, dtype=self.dtype, device=self.device)

                        weights_tensor = torch.nn.functional.interpolate(
                            weights_tensor.view(1, 1, -1), size=num_frames, mode="linear", align_corners=True
                        ).squeeze()
                    else:
                        # If only one weight, repeat it for all frames
                        weights_tensor = weights_tensor.repeat(num_frames)
                else:
                    if len(weights_tensor) < num_frames:
                        # If fewer weights than frames, repeat the last weight
                        weights_tensor = torch.cat(
                            [
                                weights_tensor,
                                torch.full(
                                    (num_frames - len(weights_tensor),),
                                    weights_tensor[-1],
                                    dtype=self.dtype,
                                    device=self.device,
                                ),
                            ]
                        )

                    # Trim if too many weights
                    if len(weights_tensor) > num_frames:
                        weights_tensor = weights_tensor[:num_frames]

                return weights_tensor

            except (ValueError, IndexError) as e:
                RESplain(f"Error parsing custom frame weights: {e}")
                return None

        return None

    def _generate_frame_weights(
        self, num_frames, dynamics, schedule, scale, is_reversed, frame_weights, step=None, custom_string=None
    ):
        # Look for the multiplier= parameter in the custom string and store it as a float value
        multiplier = None
        rate_factor = None
        start_change_factor = None
        if custom_string is not None:
            if "multiplier" in custom_string:
                multiplier_match = re.search(r"multiplier\s*=\s*([0-9.]+)", custom_string)
                if multiplier_match:
                    multiplier = float(multiplier_match.group(1))
                    # Remove the multiplier= from the custom string
                    custom_string = re.sub(r"multiplier\s*=\s*[0-9.]+", "", custom_string).strip()
                    RESplain(f"Custom multiplier detected: {multiplier}", debug=True)
            if "rate_factor" in custom_string:
                rate_factor_match = re.search(r"rate_factor\s*=\s*([0-9.]+)", custom_string)
                if rate_factor_match:
                    rate_factor = float(rate_factor_match.group(1))
                    # Remove the rate_factor= from the custom string
                    custom_string = re.sub(r"rate_factor\s*=\s*[0-9.]+", "", custom_string).strip()
                    RESplain(f"Custom rate factor detected: {rate_factor}", debug=True)
            if "start_change_factor" in custom_string:
                start_change_factor_match = re.search(r"start_change_factor\s*=\s*([0-9.]+)", custom_string)
                if start_change_factor_match:
                    start_change_factor = float(start_change_factor_match.group(1))
                    # Remove the start_change_factor= from the custom string
                    custom_string = re.sub(r"start_change_factor\s*=\s*[0-9.]+", "", custom_string).strip()
                    RESplain(f"Custom start change factor detected: {start_change_factor}", debug=True)

        if custom_string is not None and custom_string.strip() != "" and step is not None:
            custom_weights = self._generate_custom_weights(num_frames, custom_string, step)
            if custom_weights is not None:
                weights = custom_weights
                weights = torch.flip(weights, [0]) if is_reversed else weights
                return weights
            else:
                RESplain("custom frame weights failed to parse, doing the normal thing...", debug=True)

        if rate_factor is None:
            if "fast" in schedule:
                rate_factor = 0.25
            elif "slow" in schedule:
                rate_factor = 1.0
            else:  # moderate
                rate_factor = 0.5

        if start_change_factor is None:
            if "early" in schedule:
                start_change_factor = 0.0
            elif "late" in schedule:
                start_change_factor = 0.2
            else:
                start_change_factor = 0.0

        change_frames = max(round(num_frames * rate_factor), 2)
        change_start = round(num_frames * start_change_factor)
        low_value = 1.0 - scale

        if frame_weights is not None:
            weights = torch.cat([frame_weights, torch.full((num_frames,), frame_weights[-1])])
            weights = weights[:num_frames]
        else:
            if dynamics == "constant":
                weights = self._generate_constant_schedule(change_start, change_frames, low_value, num_frames)
            elif dynamics == "linear":
                weights = self._generate_linear_schedule(change_start, change_frames, low_value, num_frames)
            elif dynamics == "ease_out":
                weights = self._generate_easeout_schedule(change_start, change_frames, low_value, num_frames)
            elif dynamics == "ease_in":
                weights = self._generate_easein_schedule(change_start, change_frames, low_value, num_frames)
            elif dynamics == "middle":
                weights = self._generate_middle_schedule(change_start, change_frames, low_value, num_frames)
            elif dynamics == "trough":
                weights = self._generate_trough_schedule(change_start, change_frames, low_value, num_frames)
            else:
                raise ValueError(f"Invalid schedule: {dynamics}")

        if multiplier is None:
            multiplier = 1.0

        weights = torch.flip(weights, [0]) if is_reversed else weights
        weights = weights * multiplier
        weights = torch.clamp(weights, min=0.0, max=(max(1.0, multiplier)))
        weights = weights.to(dtype=self.dtype, device=self.device)

        return weights

    def _generate_constant_schedule(self, change_start, change_frames, low_value, num_frames):
        """constant schedule with the scale as the low weight"""
        return torch.ones(num_frames) * low_value

    def _generate_linear_schedule(self, change_start, change_frames, low_value, num_frames):
        """linear schedule from 1 to the low weight"""
        weights = torch.linspace(1, low_value, change_frames)

        weights = torch.cat([torch.full((change_start,), 1.0), weights])
        weights = torch.cat([weights, torch.full((num_frames,), weights[-1])])
        weights = weights[:num_frames]
        return weights

    def _generate_easeout_schedule(self, change_start, change_frames, low_value, num_frames, k=4.0):
        """exponential schedule from 1 to the low weight"""
        change_frames = max(change_frames, 4)
        t = torch.linspace(0, 1, change_frames, dtype=self.dtype, device=self.device)
        weights = 1.0 - (1.0 - low_value) * (1.0 - torch.exp(-k * t))
        weights = torch.cat([torch.full((change_start,), 1.0), weights])
        weights = torch.cat([weights, torch.full((num_frames,), weights[-1])])
        weights = weights[:num_frames]
        return weights

    def _generate_easein_schedule(self, change_start, change_frames, low_value, num_frames):
        """a monomial power schedule from 1 to the low weight"""
        change_frames = max(change_frames, 4)
        t = torch.linspace(0, 1, change_frames, dtype=self.dtype, device=self.device)
        weights = 1 - (1 - low_value) * torch.pow(t, 2)
        # Prepend with change_start frames of 1.0
        weights = torch.cat([torch.full((change_start,), 1.0), weights])
        total_frames_to_pad = num_frames - len(weights)
        if total_frames_to_pad > 1:
            mid_value_between_low_value_and_second_to_last_value = (weights[-2] + low_value) / 2.0
            weights[-1] = mid_value_between_low_value_and_second_to_last_value
        # Fill remaining with final value
        weights = torch.cat([weights, torch.full((num_frames,), weights[-1])])
        weights = weights[:num_frames]
        return weights

    def _generate_middle_schedule(self, change_start, change_frames, low_value, num_frames):
        """gaussian middle peaking schedule from 1 to the low weight"""

        change_frames = max(change_frames, 4)
        t = torch.linspace(0, 1, change_frames, dtype=self.dtype, device=self.device)
        weights = torch.exp(-0.5 * ((t - 0.5) / 0.2) ** 2)
        weights = weights / torch.max(weights)
        weights = low_value + (1 - low_value) * weights
        total_frames_to_pad = num_frames - len(weights)
        pad_left = total_frames_to_pad // 2
        pad_right = total_frames_to_pad - pad_left
        weights = torch.cat([torch.full((pad_left,), low_value), weights, torch.full((pad_right,), low_value)])
        if change_start > 0:
            # Pad the beginning with the first value, and truncate to num_frames
            weights = torch.cat([torch.full((change_start,), low_value), weights])
            weights = weights[:num_frames]

        return weights

    def _generate_trough_schedule(self, change_start, change_frames, low_value, num_frames):
        """
        Trough schedule with both ends at 1 and the middle at the low weight.
        When change_start > 0, creates asymmetry with shorter decay at beginning and longer at end.
        """
        change_frames = max(change_frames, 4)

        # Calculate sigma based on change_frames - controls overall decay rate
        sigma = max(0.2, change_frames / num_frames)

        if change_start == 0:
            t = torch.linspace(-1, 1, num_frames, dtype=self.dtype, device=self.device)
        else:
            asymmetry_factor = min(0.5, change_start / num_frames)

            split_point = 0.5 - asymmetry_factor

            first_size = int(split_point * num_frames)
            first_size = max(1, first_size)  # at least one frame
            t1 = torch.linspace(-1, 0, first_size, dtype=self.dtype, device=self.device)

            second_size = num_frames - first_size
            t2 = torch.linspace(0, 1, second_size, dtype=self.dtype, device=self.device)

            t = torch.cat([t1, t2])

        # shape using Gaussian function
        trough = 1.0 - torch.exp(-0.5 * (t / sigma) ** 2)

        weights = low_value + (1.0 - low_value) * trough

        return weights



