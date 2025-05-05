from methods.BaseMethod import BaseMethod
import torch.nn as nn
import torch.nn.functional as F
from libs.core import load_config
from libs.modeling import make_meta_arch


class LLMMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)
        self.video_branch = None
        self.audio_branch = None
        self.classifier = None
        self.boundary_regressor = None

    def get_model(self, cfg):
        """Initialize dual-stream model"""
        # Create video and audio branches using meta architecture
        self.video_branch = make_meta_arch(
            cfg["video_model_name"], **cfg["video_model"]
        )
        self.audio_branch = make_meta_arch(
            cfg["audio_model_name"], **cfg["audio_model"]
        )

        # Combine branches with a classifier and boundary regressor
        self.classifier = nn.Linear(cfg["combined_feature_dim"], cfg["num_classes"])
        self.boundary_regressor = nn.Linear(
            cfg["combined_feature_dim"], 2
        )  # Start and end offsets

        # Wrap with DataParallel if multiple GPUs
        if len(cfg["devices"]) > 1:
            self.video_branch = nn.DataParallel(
                self.video_branch, device_ids=cfg["devices"]
            )
            self.audio_branch = nn.DataParallel(
                self.audio_branch, device_ids=cfg["devices"]
            )
            self.classifier = nn.DataParallel(
                self.classifier, device_ids=cfg["devices"]
            )
            self.boundary_regressor = nn.DataParallel(
                self.boundary_regressor, device_ids=cfg["devices"]
            )
        else:
            self.video_branch = self.video_branch.cuda()
            self.audio_branch = self.audio_branch.cuda()
            self.classifier = self.classifier.cuda()
            self.boundary_regressor = self.boundary_regressor.cuda()

        return self.video_branch, self.audio_branch

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
            paths_cfg = load_config("configs_read_only/train_paths.yaml")
            model_cfg = load_config("configs/core_configs.yaml")
        elif mode == "valid":
            paths_cfg = load_config("configs_read_only/valid_paths.yaml")
            model_cfg = load_config("configs/core_configs.yaml")
        else:  # test mode
            paths_cfg = load_config("configs_read_only/test_paths.yaml")
            model_cfg = load_config("configs/core_configs.yaml")

        # Deep merge configs, with model_cfg taking precedence
        cfg = self.deep_merge(paths_cfg, model_cfg)

        # Set default devices if not specified
        if "devices" not in cfg:
            cfg["devices"] = ["cuda:0"]

        # Initialize model
        video_model, audio_model = self.get_model(cfg)

        # Load checkpoint for validation/test
        if mode in ["valid", "test"]:
            checkpoint_path = self.get_checkpoint_path()
            video_model = self.load_checkpoint(video_model, checkpoint_path)
            audio_model = self.load_checkpoint(audio_model, checkpoint_path)

        return video_model, audio_model, cfg
