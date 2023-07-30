import inspect
from pathlib import Path

import torch
from tensordict import TensorDict
from comfy import model_detection, model_management
from comfy.sd import CLIP, VAE, ModelPatcher, calculate_parameters, load_model_weights
from sd_meh import merge_methods
from sd_meh.merge import merge_models
from sd_meh.presets import BLOCK_WEIGHTS_PRESETS
from sd_meh.utils import weights_and_bases

base_path = Path(__file__).parent.absolute().parent.parent
models_dir = Path(base_path, "models", "checkpoints")


MERGE_METHODS = dict(inspect.getmembers(merge_methods, inspect.isfunction))
EXTS = ("ckpt", "safetensors")


def get_checkpoints():
    checkpoints = []
    for ext in EXTS:
        checkpoints.extend(list(models_dir.glob(f"**/*.{ext}")))

    return {c.stem: c for c in checkpoints}


def split_model(
    sd,
    output_clip=True,
    output_vae=True,
):
    sd_keys = sd.keys()
    clip = None
    vae = None
    model = None
    clip_target = None

    parameters = calculate_parameters(sd, "model.diffusion_model.")
    fp16 = model_management.should_use_fp16(model_params=parameters)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_detection.model_config_from_unet(
        sd, "model.diffusion_model.", fp16
    )
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type")

    offload_device = model_management.unet_offload_device()
    model = model_config.get_model(sd, "model.diffusion_model.")
    model = model.to(offload_device)
    model.load_model_weights(sd, "model.diffusion_model.")
    if output_vae:
        vae = VAE()
        w = WeightsLoader()
        w.first_stage_model = vae.first_stage_model
        load_model_weights(w, sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        clip = CLIP(clip_target, embedding_directory=None)
        w.cond_stage_model = clip.cond_stage_model
        sd = model_config.process_clip_state_dict(sd)
        load_model_weights(w, sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    return (
        ModelPatcher(
            model,
            load_device=model_management.get_torch_device(),
            offload_device=offload_device,
        ),
        clip,
        vae,
    )


class MergingExecutionHelper:
    ckpts = get_checkpoints()

    @classmethod
    def INPUT_TYPES(self):
        required = {
            "model_a": (list(self.ckpts.keys()), {"default": None}),
            "model_b": (list(self.ckpts.keys()), {"default": None}),
            "merge_mode": (list(MERGE_METHODS.keys()), {"default": "weighted_sum"}),
            "base_alpha": (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
            ),
        }
        optional = {
            "precision": ([16, 32], {"default": 16}),
            "model_c": (["None"] + list(self.ckpts.keys()), {"default": None}),
            "base_beta": (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
            ),
            "weights_alpha": ("STRING", ""),
            "weights_beta": ("STRING", ""),
            "re_basin": (["disabled", "enabled"], {"default": "disabled"}),
            "re_basin_iterations": (
                "INT",
                {"default": 10, "min": 0, "max": 1000, "step": 1},
            ),
            "weights_clip": (["disabled", "enabled"], {"default": "disabled"}),
            "device": (["cpu", "cuda"], {"default": "cpu"}),
            "work_device": (["cpu", "cuda"], {"default": "cpu"}),
            "threads": ("INT", {"default": 1, "min": 1, "step": 1}),
            "prune": (["disabled", "enabled"], {"default": "disabled"}),
            "block_weights_preset_alpha": (
                ["None"] + list(BLOCK_WEIGHTS_PRESETS.keys()),
                {"default": None},
            ),
            "presets_alpha_lambda": (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
            ),
            "block_weights_preset_alpha_b": (
                ["None"] + list(BLOCK_WEIGHTS_PRESETS.keys()),
                {"default": None},
            ),
            "block_weights_preset_beta": (
                ["None"] + list(BLOCK_WEIGHTS_PRESETS.keys()),
                {"default": None},
            ),
            "presets_beta_lambda": (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
            ),
            "block_weights_preset_beta_b": (
                ["None"] + list(BLOCK_WEIGHTS_PRESETS.keys()),
                {"default": None},
            ),
        }
        self.optional_keys = list(optional.keys())
        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "merge"
    CATEGORY = "meh"

    def merge(
        self,
        model_a,
        model_b,
        base_alpha,
        merge_mode,
        **kwargs,
    ):
        for opt_k in self.optional_keys:
            if opt_k not in kwargs:
                kwargs[opt_k] = None

        for k in ("re_basin", "weights_clip", "prune"):
            kwargs[k] = kwargs[k] == "enabled"

        for k in ("weights_alpha", "weights_beta"):
            kwargs[k] = None if kwargs[k] == "" else kwargs[k]

        for k in (
            "model_c",
            "block_weights_preset_alpha",
            "block_weights_preset_alpha_b",
            "block_weights_preset_beta",
            "block_weights_preset_beta_b",
        ):
            kwargs[k] = None if kwargs[k] == "None" else kwargs[k]

        models = {"model_a": self.ckpts[model_a], "model_b": self.ckpts[model_b]}
        if kwargs["model_c"]:
            models["model_c"] = self.ckpts[kwargs["model_c"]]

        weights, bases = weights_and_bases(
            merge_mode,
            kwargs["weights_alpha"],
            base_alpha,
            kwargs["block_weights_preset_alpha"],
            kwargs["weights_beta"],
            kwargs["base_beta"],
            kwargs["block_weights_preset_beta"],
            kwargs["block_weights_preset_alpha_b"],
            kwargs["block_weights_preset_beta_b"],
            kwargs["presets_alpha_lambda"],
            kwargs["presets_beta_lambda"],
        )

        merged = merge_models(
            models,
            weights,
            bases,
            merge_mode,
            kwargs["precision"],
            kwargs["weights_clip"],
            kwargs["re_basin"],
            kwargs["re_basin_iterations"],
            kwargs["device"],
            kwargs["work_device"],
            kwargs["prune"],
            kwargs["threads"],
        )

        if isinstance(merged, TensorDict):
            return split_model(merged.to_dict())
        return merged


NODE_CLASS_MAPPINGS = {
    "MergingExecutionHelper": MergingExecutionHelper,
}
