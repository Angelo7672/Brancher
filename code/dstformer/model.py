import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from config import Config
from keypoint_extractor import KeypointExtractorSeq
from DSTformer import DSTformer
import random


# === Utilities ===
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    """
    Usa DSTformer che prende direttamente keypoints 2D normalizzati (B,T,J,3) e restituisce 3D (B,T,J,3).
    """
    def __init__(self, device, num_joints: int, dim_feat: int, dim_rep: int, depth: int, heads: int,
                 mlp_ratio: float, drop: float, inference: bool, frame_size=(352, 640),
                 maxlen: int = 243, att_fuse: bool = True):
        """
        num_joints: numero di giunti
        dim_feat: dimensione feature
        dim_rep: dimensione rappresentazione
        depth: numero di blocchi nel DSTformer
        heads: numero di teste di attention
        mlp_ratio: pass-through al DSTformer (usato per mlp interno)
        drop: dropout rate
        frame_size: risoluzione per KeypointExtractor / Normalizer (H, W)
        maxlen: lunghezza massima sequenza (passata a DSTformer)
        att_fuse: se abilitare fusione attentiva tra stream ST/TS
        """
        super().__init__()
        self.num_joints = num_joints
        self.inference = inference
        if inference:
            self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)

        self.dst = DSTformer(
            dim_in=3,           # input per joint: (x,y,conf)
            dim_out=3,          # output per joint: (x,y,z)
            dim_feat=dim_feat,   # feature intermedia
            dim_rep=dim_rep,    # rappresentazione prima della testa
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            num_joints=num_joints,
            maxlen=maxlen,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=drop,
            attn_drop_rate=drop,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            att_fuse=att_fuse
        ).to(device)

        self.head3d = PoseHead(dim_feat=dim_feat, dim_rep=dim_rep, dim_out=3).to(device)

    def forward(self, x):
        """
        x: torch.Tensor (B, T, H, W, 3)
        return: torch.Tensor (B, T, J, 3)
        """
        # --- 1️⃣ Estrai keypoints 2D in batch ---
        if self.inference:
            kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)
        else:
            kpt_tensor = x
        
        # --- 2️⃣ Forward through DSTformer ---
        latent = self.dst(kpt_tensor)  # (B, T, J, dim_feat)

        # --- 3️⃣ Regressione 3D ---
        out3d = self.head3d(latent)  # (B, T, J, 3)

        return out3d


# ---------------------------
# Factory
# ---------------------------
def create_model(inference=False,frame_size=(352, 640)):
    cfg = Config()
    set_seed(cfg.seed)

    model = Lift3DModel(
        num_joints=cfg.num_joints,
        dim_feat=cfg.dim_feat,
        dim_rep=cfg.dim_rep,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
        inference=inference,
        frame_size=frame_size,  #(H, W)
        device=cfg.device
    ).to(cfg.device)

    return model