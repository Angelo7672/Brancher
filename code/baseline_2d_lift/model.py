import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from config import Config
from keypoint_extractor import KeypointExtractorSeq
import random


# === Utilities ===
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ==== Embedding 2D keypoints ===
class Kpt2DEncoder(nn.Module):
    """
    MLP per convertire i keypoints 2D (J,3) in embedding (embed_dim)
    """
    def __init__(self, num_joints, in_dim=3, embed_dim=512, drop: float = 0.0):
        """
        num_joints: numero di giunti (es. COCO = 17)
        in_dim: dimensione input (x,y) → 3
        embed_dim: dimensione embedding per il modello 3D
        """
        super().__init__()
        self.num_joints = num_joints
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.fc = nn.Sequential(
            nn.Linear(num_joints * in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        """
        x: (B, T, J, 3) keypoints normalizzati
        ritorna: (B, T, embed_dim)
        """
        B, T, J, D = x.shape
        assert J == self.num_joints and D == self.in_dim, \
            f"Expected ({self.num_joints},{self.in_dim}), got ({J},{D})"

        x = x.view(B, T, J * D)  # flatten joints per frame
        # process per-frame feature vector (batch-first MLP)
        # fc expects (B*T, J*2) or it will broadcast over last dim; apply reshape:
        x = x.view(B * T, J * D)
        x = self.fc(x)  # (B*T, embed_dim)
        x = x.view(B, T, self.embed_dim)
        return x


# === Trasformer temporale ===
class TemporalTransformer(nn.Module):
    """
    Questa è la parte della rete che cattura relazioni temporali tra i frame della sequenza di pose.
    """
    def __init__(self, d_model: int, depth: int, nhead: int, mlp_ratio: float, drop: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=drop,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.pos(x)
        x = self.encoder(x)
        return x


# === MLP ===
class PoseHead(nn.Module):
    def __init__(self, d_model: int, num_joints: int, drop: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model * 2, num_joints * 3),
        )
        self.num_joints = num_joints

    def forward(self, x):
        # x: (B, T, D)
        y = self.mlp(x)  # (B, T, J*3)
        B, T = x.shape[:2]
        y = y.view(B, T, self.num_joints, 3)
        return y


# === Lift3DModel ===
class Lift3DModel(nn.Module):
    def __init__(self, device, num_joints: int, d_model: int, depth: int, heads: int,
                 mlp_ratio: float, drop: float, frame_size=(640, 416)):
        """
        frame_size: (H, W) usata per il normalizer; passare la risoluzione dei frames in input
        """
        super().__init__()
        self.num_joints = num_joints
        self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, 
                                                       frame_size=frame_size,
                                                       device=device)
        self.encoder2d = Kpt2DEncoder(num_joints=num_joints,
                                      in_dim=3,   # (x,y,conf)
                                      embed_dim=d_model,
                                      drop=drop)
        self.temporal = TemporalTransformer(d_model=d_model,
                                            depth=depth,
                                            nhead=heads,
                                            mlp_ratio=mlp_ratio,
                                            drop=drop)
        self.head3d = PoseHead(d_model=d_model,
                               num_joints=num_joints,
                               drop=drop)

    def forward(self, x):
        """
        x: (B,T,H,W,3).
        Output: torch.Tensor (B,T,J,3)
        """
        # --- 1️⃣ Estrai keypoints 2D in batch ---
        kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)

        # --- 2️⃣ Embedding ---
        enc = self.encoder2d(kpt_tensor) # (B,T,embed_dim)

        # --- 3️⃣ Temporal Transformer ---
        tmp = self.temporal(enc)  # (B,T,embed_dim)

        # --- 4️⃣ Regressione 3D ---
        out3d = self.head3d(tmp)  # (B,T,J,3)

        return out3d

# ---------------------------
# Factory
# ---------------------------
def create_model():
    cfg = Config()
    set_seed(cfg.seed)

    model = Lift3DModel(
        num_joints=cfg.num_joints,
        device=cfg.device,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
        frame_size=cfg.frame_size  #(H, W)
    ).to(cfg.device)

    summary(
        model,
        input_size=(cfg.batch_size, cfg.clip_len, cfg.frame_size[0], cfg.frame_size[1], 3),    # (B, T, W, H, 3)
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        device=cfg.device,
    )

    return model