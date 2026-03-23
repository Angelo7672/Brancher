from dataclasses import dataclass
import torch

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    pe_size: int = (256, 192)  # Per ViTPose
    clip_len: int = 81
    num_joints: int = 17

    # Keypoint Extractor
    config_file: str = "AR17/code/keypoints_module/vitpose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
    checkpoint_file: str = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
    yolo_version: str = "./yolo12s.pt" # small model 9.3M parameters, nano model 2.6M parameters

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"