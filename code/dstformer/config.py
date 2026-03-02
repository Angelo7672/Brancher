from dataclasses import dataclass
import torch


# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    datapath_sport: str = "/mnt/proj2/eu-25-52/ar17/SportsPose_reprocessed_v3"
    datapath_air: str = "/Experiments/angeloa00/AirPose_COCO/train_flipped"
    base_dir: str = "experiments/dstformer"
    frame_size_sport: int = (640, 416)  # Per SportsPose
    frame_size_air: int = (352, 640) # Per AirPose
    pe_size: int = (256, 192)  # Per ViTPose
    clip_len: int = 243
    num_joints: int = 17
    scale_factor: float = 2.16   # meters
    pretrained_weights: str = ""

    # Keypoint Extractor
    config_file: str = "AR17/code/keypoints_module/vitpose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
    checkpoint_file: str = ".cache/torch/hub/checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
    #checkpoint_file: str = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-216eae50_20230314.pth"
    yolo_version: str = "./yolo12s.pt" # small model 9.3M parameters, nano model 2.6M parameters

    # Model
    dim_feat: int = 512
    dim_rep: int = 512                        
    transf_depth: int = 5
    transf_heads: int = 8
    transf_mlp_ratio: int = 2
    drop: float = 0.0

    # Train
    epochs: int = 60
    batch_size: int = 8
    lr: float = 0.0005
    weight_decay: float = 0.01
    num_workers: int = 4

    # Loss
    velocity_loss_weight: float = 20.0

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"