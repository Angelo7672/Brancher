from dataclasses import dataclass
import torch

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    train_index: str = "data/train_index.json"  # JSON list of samples
    val_index: str = "data/val_index.json"
    frames_root: str = "data/frames"           # contains subfolders with extracted frames per clip
    labels_root: str = "data/labels"           # contains parallel .npz label files with 3D joints
    img_size: int = 256
    clip_len: int = 27                          # temporal window length (odd, so center = T//2)
    frame_stride: int = 1                       # frame step inside a clip

    # Model
    num_joints: int = 17                        # change to your skeleton
    embed_dim: int = 512                        # from resnet18 penultimate layer (512)
    transf_depth: int = 6
    transf_heads: int = 8
    transf_mlp_ratio: float = 4.0
    drop: float = 0.1

    # Train
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    num_workers: int = 8
    amp: bool = True
    grad_clip: float = 1.0

    # Loss weights
    lambda_mpjpe: float = 1.0
    lambda_bone: float = 0.1

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"