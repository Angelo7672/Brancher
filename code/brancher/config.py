from dataclasses import dataclass
import torch

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    datapath_athlete: str = "/Experiments/angeloa00/AthletePose3D/frame_81"
    datapath_drone: str = "/mnt/proj2/eu-25-52/ar17/DroneDataset/train"
    datapath_air: str = "/mnt/proj2/eu-25-52/ar17/AirPose_H36M/train"
    base_dir: str = "/Experiments/angeloa00/brancher"
    frame_size_air: int = (352, 640) # Per AirPose
    pe_size: int = (256, 192)  # Per ViTPose
    clip_len: int = 81
    num_joints: int = 17
    num_bones: int = 16
    scale_factor: float = 2.0
    pretrained_weights: str = "motionagformer-s-h36m.pth.tr"
    weights: str = "experiments/brancher/model_20260223-231757/best_weight.pth"

    # Keypoint Extractor
    config_file: str = "AR17/code/keypoints_module/vitpose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
    #checkpoint_file: str = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-216eae50_20230314.pth"
    checkpoint_file: str = ".cache/torch/hub/checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
    yolo_version: str = "./yolo12s.pt" # small model 9.3M parameters, nano model 2.6M parameters

    # Model
    n_layers: int = 26
    dim_in: int = 3 
    dim_feat: int = 64
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
    epochs: int = 60
    patience: int = 55
    batch_size: int = 32
    lr: float = 0.0005
    weight_decay: float = 0.01
    warmup_epochs: int = 0
    num_workers: int = 4

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"