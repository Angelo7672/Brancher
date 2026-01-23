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

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    datapath: str = "/mnt/proj2/eu-25-52/ar17/SportsPose_reprocessed_v3"
    base_dir: str = "experiments/brancher"
    frame_size: int = (640, 416)  # Per YOLO
    pe_size: int = (256, 192)  # Per ViTPose
    clip_len: int = 243
    num_joints: int = 17
    num_bones: int = 13
    scale_factor: float = 2.16   # meters

    # Keypoint Extractor
    config_file: str = "AR17/code/keypoints_module/vitpose/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
    checkpoint_file: str = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
    yolo_version: str = "./yolo12s.pt" # small model 9.3M parameters, nano model 2.6M parameters

    # Model
    n_layers: int = 16
    dim_in: int = 3 
    dim_feat: int = 128
    dim_rep: int = 512
    dim_out: int = 3
    mlp_ratio: int = 4
    act_layer: str = 'gelu'
    attn_drop: float = 0.0
    drop_path: float = 0.0
    drop: float = 0.0
    use_layer_scale: bool = True
    layer_scale_init_value: float = 0.00001
    use_adaptive_fusion: bool = True
    num_heads: int = 8
    qkv_bias: bool = False
    qkv_scale: None = None
    hierarchical: bool = False
    use_temporal_similarity: bool = True # Only useful when use_tcn = False
    neighbour_num: int = 2  # Only useful when use_temporal_similarity = True
    temporal_connection_len: int = 1 # Only useful when use_tcn = False, use_temporal_similarity = False
    use_tcn: bool = False
    graph_only: bool = False

    # Train
    epochs: int = 100
    patience: int = 10
    batch_size: int = 8
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