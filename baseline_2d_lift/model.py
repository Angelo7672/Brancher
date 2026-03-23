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
    """
    Original PE from Attention is All You Need
    """
    def __init__(self, d_model, max_seq_len=81):
        super().__init__()
        self.d_model = d_model
        
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # Stesso device del modello

    def forward(self, x):
        # x: (B, T, D) con batch_first=True
        seq_len = x.size(1)
        pe = self.pe[:seq_len].unsqueeze(0)  # (1, T, D)
        x = x * (self.d_model ** 0.5) + pe
        return x

# === Temporal Module ===
class TemporalTransformer(nn.Module):
    """
    This is the part of the network that captures the temporal relationships between frames in the pose sequence.
    Only Temporal Attention, not Spatial.
    """
    def __init__(self, d_model: int, depth: int, nhead: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.pos(x)
        x = self.encoder(x)
        return x


# === MLP ===
class PoseHead(nn.Module):
    def __init__(self, dim_feat, dim_rep, dim_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_feat, dim_rep),
            nn.Tanh(),
            nn.Linear(dim_rep, dim_out)
        )

    def forward(self, x):
        return self.mlp(x)


# === Lift3DModel ===
class Lift3DModel(nn.Module):
    def __init__(self, device, num_joints, d_model, depth, heads, inference, frame_size):
        super().__init__()
        self.num_joints = num_joints
        self.inference = inference
        if inference:
            self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)
        self.encoder2d = nn.Linear(num_joints * 3, d_model) # PE works on the T dimension, tokens are the frames, not the joints
        self.temporal = TemporalTransformer(d_model=d_model, depth=depth, nhead=heads).to(device)
        self.head3d = PoseHead(dim_feat=d_model, dim_rep=d_model, dim_out=num_joints*3).to(device)

    def forward(self, x):
        """
        Input: torch.Tensor (B,T,H,W,3)
        Output: torch.Tensor (B,T,J,3)
        """
        # --- 1️⃣ Extract 2D keypoints in batches ---
        if self.inference:
            kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)
        else:
            kpt_tensor = x  # Input just in shape (B,T,J,3)

        # --- 2️⃣ Embedding ---
        B, T, J, D = kpt_tensor.shape
        kpt_flat = kpt_tensor.reshape(B, T, J * D)  # (B, T, J*3)
        enc = self.encoder2d(kpt_flat) # (B,T,embed_dim)

        # --- 3️⃣ Temporal Transformer ---
        tmp = self.temporal(enc)  # (B,T,embed_dim)

        # --- 4️⃣ 3D Regression ---
        out3d = self.head3d(tmp)  
        out3d = out3d.reshape(B, T, self.num_joints, 3) # (B,T,J,3)

        return out3d


# === Factory ===
def create_model(inference = False, frame_size=(352, 640)):
    cfg = Config()
    set_seed(cfg.seed)

    model = Lift3DModel(
        num_joints=cfg.num_joints,
        device=cfg.device,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        inference=inference,
        frame_size=frame_size  #(H, W)
    ).to(cfg.device)

    return model


# === Optuna optimizer factory ===
def optimize_model(depth: int):
    cfg = Config()

    model = Lift3DModel(
        num_joints=17,
        device=cfg.device,
        d_model=cfg.embed_dim,
        depth=depth,
        heads=cfg.transf_heads,
        frame_size=(352, 640)  #(H, W)
    ).to(cfg.device)

    return model