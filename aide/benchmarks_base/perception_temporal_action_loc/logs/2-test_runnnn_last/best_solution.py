from methods.BaseMethod import BaseMethod
import torch.nn as nn
from libs.core import load_config
from libs.modeling import make_meta_arch


class LLMMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def get_model(self, cfg):
        """Initialize Dual-Branch ActionFormer model"""
        video_model = make_meta_arch(cfg["video_model_name"], **cfg["video_model"])
        audio_model = make_meta_arch(cfg["audio_model_name"], **cfg["audio_model"])

        # Wrap with DataParallel if multiple GPUs
        if len(cfg["devices"]) > 1:
            video_model = nn.DataParallel(video_model, device_ids=cfg["devices"])
            audio_model = nn.DataParallel(audio_model, device_ids=cfg["devices"])
        else:
            video_model = video_model.cuda()
            audio_model = audio_model.cuda()

        return video_model, audio_model

    def deep_merge(self, dict1, dict2):
        result = dict1.copy()
        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def run(self, mode):
        """Handle different running modes"""

        # Load appropriate configs
        if mode == "train":
            paths_cfg = load_config("input/configs_read_only/train_paths.yaml")
            model_cfg = load_config("input/configs/core_configs.yaml")
        elif mode == "valid":
            paths_cfg = load_config("input/configs_read_only/valid_paths.yaml")
            model_cfg = load_config("input/configs/core_configs.yaml")
        else:  # test mode
            paths_cfg = load_config("input/configs_read_only/test_paths.yaml")
            model_cfg = load_config("input/configs/core_configs.yaml")

        # Deep merge configs, with model_cfg taking precedence
        cfg = self.deep_merge(paths_cfg, model_cfg)

        # Set default devices if not specified
        if "devices" not in cfg:
            cfg["devices"] = ["cuda:0"]

        # Initialize models
        video_model, audio_model = self.get_model(cfg)

        # Load checkpoint for validation/test
        if mode in ["valid", "test"]:
            checkpoint_path = self.get_checkpoint_path()
            video_model = self.load_checkpoint(video_model, checkpoint_path)
            audio_model = self.load_checkpoint(audio_model, checkpoint_path)

        return video_model, audio_model
