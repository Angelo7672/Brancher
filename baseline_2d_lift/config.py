from dataclasses import dataclass
import torch

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    datapath_athlete: str = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/pose_3d_v3/frame_81"
    datapath_air: str = "/Experiments/angeloa00/AirPose_COCO/train_flipped"
    base_dir: str = "experiments/baseline_2d_lift"
    pe_size: int = (256, 192)  # Per ViTPose
    clip_len: int = 81
    num_joints: int = 17
    pretrained_weights: str = "/Experiments/angeloa00/baseline_2d_lift/model_20260204-221245/best_weight.pth"

    # Keypoint Extractor
    config_file: str = "keypoints_module/vitpose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
    checkpoint_file: str = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
    #checkpoint_file: str = ".cache/torch/hub/checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
    yolo_version: str = "./yolo12s.pt" # small model 9.3M parameters, nano model 2.6M parameters

    # Model
    embed_dim: int = 64                        
    transf_depth: int = 3
    transf_heads: int = 8

    # Train
    epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    lr: float = 0.0005
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    num_workers: int = 4

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
