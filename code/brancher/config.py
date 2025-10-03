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
    img_size: int = (256, 192)               # height, width                         
    clip_len: int = 243                         # 243 frames

    # Model
    num_joints: int = 17                        # SportsPose usa 17 joints
    maxlen: int = 243
    embed_dim: int = 512
    transf_mlp_ratio: int = 2
    transf_depth: int = 5
    dim_rep: int = 512
    transf_heads: int = 8
    att_fuse: bool = True

    # Train
    epochs: int = 30
    batch_size: int = 2 # 1 = 7 views
    lr: float = 0.0005
    weight_decay: float = 0.01
    lr_decay: float = 0.99
    warmup_epochs: int = 2
    num_workers: int = 2
    amp: bool = True
    grad_clip: float = 1.0
    drop: float = 0.0

    # Loss weights
    lambda_bone: float = 0.1

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Skeleton
    bones: list = None

    def __post_init__(self):
        # aggiunge automaticamente le bones al config
        self.bones = BONES