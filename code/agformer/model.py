import torch
import torch.nn as nn
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
    Usa AGFormer che prende direttamente keypoints 2D normalizzati (B,T,J,3) e restituisce 3D (B,T,J,3).
    """
    def __init__(self, device, num_joints, n_layers, dim_in, dim_feat, dim_rep, dim_out,
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
        self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)

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

        self.head3d = PoseHead(dim_feat=dim_feat, dim_rep=dim_rep, dim_out=3).to(device)

    def forward(self, x):
        """
        x: torch.Tensor (B, T, H, W, 3)
        return: torch.Tensor (B, T, J, 3)
        """
        # --- 1️⃣ Estrai keypoints 2D in batch ---
        kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)
        #kpt_tensor = x
        
        # --- 2️⃣ Forward through AGFormer ---
        latent = self.ag(kpt_tensor)  # (B, T, J, dim_feat)

        # --- 3️⃣ Regressione 3D ---
        out3d = self.head3d(latent)  # (B, T, J, 3)

        return out3d


# ---------------------------
# Factory
# ---------------------------
def create_model():
    cfg = Config()
    set_seed(cfg.seed)

    model = Lift3DModel(
        num_joints=cfg.num_joints,
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
        frame_size=cfg.frame_size,  #(H, W)
        device=cfg.device
    ).to(cfg.device)

    return model