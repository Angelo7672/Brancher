from dataclasses import dataclass
import torch

# -------------------------------
# Skeleton definition (SportsPose)
# -------------------------------
SPORTSPOSE_KINEMATIC_TREE = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [5, 3],
    [6, 4],
    [3, 1],
    [4, 2],
    [1, 0],
    [2, 0],
]

# Converto nel formato usato dalla loss
BONES = [(i, j) for (i, j) in SPORTSPOSE_KINEMATIC_TREE]

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    datapath: str = "/Experiments/angeloa00/SportsPose_reprocessed_v3"
    #datapath: str = "/DATA/SportsPose"
    base_dir: str = "/Experiments/angeloa00/baseline_2d_lift"
    frame_size: int = (640, 416)  # Per YOLO
    pe_size: int = (256, 192)  # Per ViTPose
    clip_len: int = 243
    num_joints: int = 17
    scale_factor: float = 2.5   # meters, candidati 2.0 e 2.5

    # Keypoint Extractor
    config_file: str = "AR17/code/keypoints_module/vitpose/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
    checkpoint_file: str = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
    yolo_version: str = "./yolo12s.pt" # small model 9.3M parameters, nano model 2.6M parameters

    # Model
    embed_dim: int = 512                        
    transf_depth: int = 5
    transf_heads: int = 8

    # Train
    epochs: int = 1000
    patience: int = 10
    batch_size: int = 32
    lr: float = 0.0005
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    num_workers: int = 4

    # Loss
    bone_loss_weight: float = 0.2
    velocity_loss_weight: float = 0.1
    acceleration_loss_weight: float = 0.1

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"