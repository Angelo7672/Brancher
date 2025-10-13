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
    datapath: str = "/DATA/SportsPose"
    frame_size: int = (576, 360)          #(640, 400)  # Per YOLOv8
    pe_size: int = (256, 192)  # Per ViTPose
    clip_len: int = 210
    num_joints: int = 17
    root_rel: bool = False

    # Model
    maxlen: int = 243
    embed_dim: int = 512
    transf_mlp_ratio: int = 2
    transf_depth: int = 5
    dim_rep: int = 512
    transf_heads: int = 8
    att_fuse: bool = True

    # Train
    epochs: int = 30
    batch_size: int = 1
    lr: float = 0.0005
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    num_workers: int = 2
    amp: bool = True
    grad_clip: float = 1.0
    drop: float = 0.0

    # Loss
    lambda_3d_velocity: float = 20.0
    lambda_scale: float = 0.5
    lambda_lv: float = 0.0
    lambda_lg: float = 0.0
    lambda_a: float = 0.0
    lambda_av: float = 0.0
    lambda_2d_proj: float = 0.0

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bones: list = None

    def __post_init__(self):
        self.bones = BONES