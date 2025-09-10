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
    #train_index: str = "data/train_index.json"  # JSON list of samples
    #val_index: str = "data/val_index.json"
    #frames_root: str = "data/frames"           # contains subfolders with extracted frames per clip
    #labels_root: str = "data/labels"           # contains parallel .npz label files with 3D joints
    img_size: int = 256
    clip_len: int = 20                          # temporal window length (odd, so center = T//2)
    #frame_stride: int = 1                       # frame step inside a clip

    # Model
    num_joints: int = 17                        # SportsPose usa 17 joints
    embed_dim: int = 512                        # from resnet18 penultimate layer (512)
    transf_depth: int = 6
    transf_heads: int = 8
    transf_mlp_ratio: float = 4.0
    drop: float = 0.1

    # Train
    epochs: int = 1
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    num_workers: int = 2
    amp: bool = True
    grad_clip: float = 1.0

    # Loss weights
    lambda_mpjpe: float = 1.0
    lambda_bone: float = 0.1

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Skeleton
    bones: list = None

    def __post_init__(self):
        # aggiunge automaticamente le bones al config
        self.bones = BONES