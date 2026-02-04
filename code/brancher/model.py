import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
from config import Config
from keypoint_extractor import KeypointExtractorSeq
from MotionAGFormer import MotionAGFormer
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


# === Rotation Head ===
class RotationHead(nn.Module):
    def __init__(self, dim_feat, num_joints=17, num_bones=16):
        super().__init__()
        self.num_bones = num_bones
        # Proiettiamo dal d_model del transformer a 6 dimensioni per ogni osso
        self.fc = nn.Linear(dim_feat * num_joints, num_bones * 6)
    
    def compute_rotation_matrix_from_6d(self, custom_6d):
        """
        Versione ottimizzata e vettorizzata del processo Gram-Schmidt.
        Input: [Batch, Temp, Bones, 6]
        Output: [Batch, Temp, Bones, 3, 3] (Matrici di rotazione SO(3))
        """
        batch_size, seq_len, num_bones, _ = custom_6d.shape
        x_raw = custom_6d.view(-1, 6) # Appiattiamo per facilitare i calcoli

        x = x_raw[:, :3] # Primo vettore (a1)
        y = x_raw[:, 3:] # Secondo vettore (a2)

        # Normalizzazione del primo asse
        x_axis = F.normalize(x, p=2, dim=-1)

        # Rendiamo il secondo asse ortogonale al primo e normalizziamo
        # (a2 - proj_x1(a2))
        z_axis = torch.cross(x_axis, y, dim=-1)
        z_axis = F.normalize(z_axis, p=2, dim=-1)

        # Il terzo asse è il prodotto vettoriale degli altri due
        y_axis = torch.cross(z_axis, x_axis, dim=-1)

        # Componiamo la matrice [3x3]
        # Stack come colonne: [x_axis | y_axis | z_axis]
        res = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        return res.view(batch_size, seq_len, num_bones, 3, 3)

    def forward(self, x):
        B, T, J, D = x.shape
        x = x.reshape(B, T, J * D) # Flatten dei giunti
        x = self.fc(x)
        x = x.view(B, T, self.num_bones, 6)
        x = self.compute_rotation_matrix_from_6d(x)

        return x


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
        # Appiattiamo Batch, Tempo e Giunti in un'unica dimensione per la batch inference
        x = x.view(-1, D) # [B*T*J, D]
        
        # 1. Proiezione lineare e reshaping per le deconv
        x = self.fc(x) # [B*T*J, base_channels * 8 * 6]
        x = x.view(-1, self.base_channels, self.init_h, self.init_w)
        
        # 2. Upsampling tramite deconvolutional layers
        x = self.deconv(x) # [B*T*J, 1, 64, 48]
        
        # 3. Ripristino delle dimensioni originali
        return x.view(B, T, J, self.H, self.W)


# === Brancher ===
class Brancher(nn.Module):
    def __init__(self, device, num_joints, num_bones, n_layers, dim_in, dim_feat, dim_rep, dim_out,
                 mlp_ratio, act_layer, attn_drop, drop, drop_path,
                 use_layer_scale, layer_scale_init_value, use_adaptive_fusion,
                 num_heads, qkv_bias, qkv_scale, hierarchical,
                 use_temporal_similarity, temporal_connection_len,
                 use_tcn, graph_only, neighbour_num,
                 n_frames, frame_size):
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
        self.num_bones = num_bones
        #self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)

        act_mapper = {
            "gelu": nn.GELU,
            "relu": nn.ReLU
        }

        self.ag = MotionAGFormer(n_layers=n_layers,
                               dim_in=dim_in,
                               dim_feat=dim_feat,
                               dim_rep=dim_rep,
                               dim_out=dim_out,
                               mlp_ratio=mlp_ratio,
                               act_layer=act_mapper[act_layer],
                               attn_drop=attn_drop,
                               drop=drop,
                               drop_path=drop_path,
                               use_layer_scale=use_layer_scale,
                               layer_scale_init_value=layer_scale_init_value,
                               use_adaptive_fusion=use_adaptive_fusion,
                               num_heads=num_heads,
                               qkv_bias=qkv_bias,
                               qkv_scale=qkv_scale,
                               hierarchical=hierarchical,
                               num_joints=num_joints,
                               use_temporal_similarity=use_temporal_similarity,
                               temporal_connection_len=temporal_connection_len,
                               use_tcn=use_tcn,
                               graph_only=graph_only,
                               neighbour_num=neighbour_num,
                               n_frames=n_frames
                            ).to(device)

        # Branch 1: 3D Pose
        self.head3d = PoseHead(dim_feat=dim_feat, dim_rep=dim_rep, dim_out=3).to(device)

        # Branch 2: Rotations
        self.rotation_head = RotationHead(dim_feat=dim_feat, num_joints=num_joints, num_bones=num_bones).to(device)

        # Branch 3: Uncertainty 
        self.uncertainty_head = UncertaintyHead(dim_feat=dim_feat).to(device)


    def forward(self, x):
        """
        x: torch.Tensor (B, T, H, W, 3)
        return: torch.Tensor (B, T, J, 3)
        """
        # --- Estrai keypoints 2D in batch ---
        #kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)
        kpt_tensor = x
        
        # --- Forward through AGFormer ---
        latent = self.ag(kpt_tensor)  # (B, T, J, dim_feat)

        # --- Pose 3D ---
        out3d = self.head3d(latent)  # (B, T, J, 3)

        # --- Rotations ---
        rotations = self.rotation_head(latent)  # (B, T, bones, 3, 3)

        # --- Heatmap ---
        uncertainty = self.uncertainty_head(latent)  # (B, T, J, H, W)

        return out3d, rotations, uncertainty


# ---------------------------
# Factory
# ---------------------------
def create_model(dataset_name: str):
    cfg = Config()
    set_seed(cfg.seed)

    model = Brancher(
        num_joints=cfg.num_joints,
        num_bones=cfg.num_bones,
        n_layers=cfg.n_layers,
        dim_in=cfg.dim_in,
        dim_feat=cfg.dim_feat,
        dim_rep=cfg.dim_rep,
        dim_out=cfg.dim_out,
        mlp_ratio=cfg.mlp_ratio,
        act_layer=cfg.act_layer,
        attn_drop=cfg.attn_drop,
        drop=cfg.drop,
        drop_path=cfg.drop_path,
        use_layer_scale=cfg.use_layer_scale,
        layer_scale_init_value=cfg.layer_scale_init_value,
        use_adaptive_fusion=cfg.use_adaptive_fusion,
        num_heads=cfg.num_heads,
        qkv_bias=cfg.qkv_bias,
        qkv_scale=cfg.qkv_scale,
        hierarchical=cfg.hierarchical,
        use_temporal_similarity=cfg.use_temporal_similarity,
        temporal_connection_len=cfg.temporal_connection_len,
        use_tcn=cfg.use_tcn,
        graph_only=cfg.graph_only,
        neighbour_num=cfg.neighbour_num,
        n_frames=cfg.clip_len,
        frame_size=cfg.frame_size_air if dataset_name=="air" else cfg.frame_size_sport,  #(H, W)
        device=cfg.device
    ).to(cfg.device)

    return model