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


# === 3D Head ===
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

# === Uncertainty Head ===
class UncertaintyHead(nn.Module):
    def __init__(self, dim_feat, heatmap_size=(64, 48)):
        super().__init__()
        self.H, self.W = heatmap_size
        self.init_h, self.init_w = 8, 6
        self.base_channels = 128  # Definiamo una variabile per coerenza
        
        self.fc = nn.Sequential(
            nn.Linear(dim_feat, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.base_channels * self.init_h * self.init_w)
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels, 64, 4, 2, 1),  # 8x6 -> 16x12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),                 # 16x12 -> 32x24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)                  # 32x24 -> 64x48
        )

    def forward(self, x):
        B, T, J, D = x.shape
        x_flat = x.view(-1, D)  # Es: 33048 campioni
        
        # Specifichiamo una dimensione del chunk piccola
        # Processando 1024 giunti alla volta, il consumo di memoria istantaneo crolla
        chunk_size = 1024 
        output_list = []
        
        for i in range(0, x_flat.size(0), chunk_size):
            chunk = x_flat[i : i + chunk_size]
            
            # Passaggio nel blocco lineare
            c = self.fc(chunk)
            c = c.view(-1, self.base_channels, self.init_h, self.init_w)
            
            # Passaggio nelle deconv
            c = self.deconv(c)
            
            # Spostiamo il chunk su CPU se la memoria è davvero critica (opzionale)
            # output_list.append(c.cpu()) 
            output_list.append(c)
            
        # Ricostruiamo il tensore finale
        heatmap = torch.cat(output_list, dim=0)
        return heatmap.view(B, T, J, self.H, self.W)


# === Brancher ===
class Brancher(nn.Module):
    """
    Usa DSTformer che prende direttamente keypoints 2D normalizzati (B,T,J,3)
    e restituisce 3D (B,T,J,3).
    """
    def __init__(self, device, num_joints: int, dim_feat: int, dim_rep: int, depth: int, heads: int,
                 mlp_ratio: float, drop: float, frame_size=(256, 192),
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
        #self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)

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

        # Branch 1: 3D Pose
        self.head3d = PoseHead(dim_feat=dim_feat, dim_rep=dim_rep, dim_out=3).to(device)

        # Branch 3: Uncertainty 
        self.uncertainty_head = UncertaintyHead(dim_feat=dim_feat).to(device)


    def forward(self, x):
        """
        x: torch.Tensor (B, T, H, W, 3)
        return: torch.Tensor (B, T, J, 3)
        """
        # --- 1️⃣ Estrai keypoints 2D in batch ---
        #kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)
        kpt_tensor = x
        
        # --- 2️⃣ Forward through DSTformer ---
        latent = self.dst(kpt_tensor)  # (B, T, J, dim_feat)

        # --- 3️⃣ Regressione 3D ---
        out3d = self.head3d(latent)  # (B, T, J, 3)

        # --- Creazione Heatmap ---
        uncertainty = self.uncertainty_head(latent)  # (B, T, J, H, W)

        return out3d, uncertainty


# ---------------------------
# Factory
# ---------------------------
def create_model():
    cfg = Config()
    set_seed(cfg.seed)

    model = Brancher(
        num_joints=cfg.num_joints,
        dim_feat=cfg.dim_feat,
        dim_rep=cfg.dim_rep,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
        frame_size=cfg.frame_size,  #(H, W)
        device=cfg.device
    ).to(cfg.device)

    return model