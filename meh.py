import logging
from sd_meh.merge import merge_models, save_model
import inspect
from sd_meh import merge_methods
from sd_meh.presets import BLOCK_WEIGHTS_PRESETS
from sd_meh.utils import weights_and_bases
from pathlib import Path

logging.basicConfig(format="%(levelname)s: %(message)s", level="DEBUG")

base_path = Path(__file__).parent.absolute().parent.parent
models_dir = Path(base_path, "models", "checkpoints")


MERGE_METHODS = dict(inspect.getmembers(merge_methods, inspect.isfunction))
EXTS = ("ckpt", "safetensors")


def get_checkpoints():
    checkpoints = []
    for ext in EXTS:
        checkpoints.extend(list(models_dir.glob(f"**/*.{ext}")))

    return {c.stem: c for c in checkpoints}


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
            "merge_name": ("STRING", {"default": "model_out"}),
            "output_format": (["safetensors", "ckpt"], {"default": "safetensors"}),
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
            "device": (["cpu", "gpu"], {"default": "cpu"}),
            "work_device": (["cpu", "gpu"], {"default": "cpu"}),
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

    RETURN_TYPES = ()
    OUTPUT_NODE = True
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

        save_model(merged, Path(models_dir, kwargs["merge_name"]), kwargs["output_format"])

        return {}


NODE_CLASS_MAPPINGS = {
    "MergingExecutionHelper": MergingExecutionHelper,
}
