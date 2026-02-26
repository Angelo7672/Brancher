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
        # Project from the d_model of the 6-dimensional transformer for each bone
        self.fc = nn.Linear(dim_feat * num_joints, num_bones * 6)
        self.tanh = nn.Tanh()
    
    def compute_rotation_matrix_from_6d(self, custom_6d):
        """
        Optimized and vectorized version of the Gram-Schmidt process.
        Input: [Batch, Temp, Bones, 6]
        Output: [Batch, Temp, Bones, 3, 3] (Rotation Matrix in SO(3))
        """
        batch_size, seq_len, num_bones, _ = custom_6d.shape
        x_raw = custom_6d.view(-1, 6)

        a = x_raw[:, :3] # First vector
        b = x_raw[:, 3:] # Second vector

        # --- Normalization of the first axis ---
        x_axis = F.normalize(a, p=2, dim=-1)

        # --- Let's make the second axis orthogonal to the first and normalize ---
        dot = torch.sum(x_axis * b, dim=-1, keepdim=True)
        y_axis = F.normalize(b - dot * x_axis, p=2, dim=-1) # b - (r1^T * b)*r1 / ||b - (r1^T * b)*r1||

        # --- The third axis is the vector product of the other two ---
        z_axis = torch.cross(x_axis, y_axis, dim=-1)

        # Let's compose the matrix [3x3]
        # Stack as columns: [x_axis | y_axis | z_axis]
        res = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        return res.view(batch_size, seq_len, num_bones, 3, 3)

    def forward(self, x):
        B, T, J, D = x.shape
        x = x.reshape(B, T, J * D) # Joints flatten
        x = self.fc(x)
        x = self.tanh(x)
        x = x.view(B, T, self.num_bones, 6)
        x = self.compute_rotation_matrix_from_6d(x)

        return x


# === Uncertainty Head ===
class UncertaintyHead(nn.Module):
    def __init__(self, dim_feat, dim_rep, heatmap_size=(64, 48)):
        super().__init__()
        self.H, self.W = heatmap_size
        # init_h, init_w: These are the "embryonic" dimensions of the heatmap.
        # We can't go from a vector to 64x48 in one go. We start with a small grid and gradually "enlarge" it.
        self.init_h, self.init_w = 8, 6
        # base_channels: This is the initial "depth" (number of filters).
        # Imagine that the Linear layer creates 32 small 8x6 images, each containing different pose information.
        self.base_channels = 32 
        
        self.fc = nn.Sequential(
            nn.Linear(dim_feat, dim_rep),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.heatmap_fc = nn.Sequential( 
            nn.Linear(dim_rep, self.base_channels * self.init_h * self.init_w),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

        # Joint-wise Uncertainty Branch
        # Produces a log_var value for each joint
        self.uncertainty_fc = nn.Linear(dim_rep, 1)

    def forward(self, x):
        B, T, J, D = x.shape
        x_flat = x.view(-1, D) 
        
        feat = self.fc(x_flat) # [B*T*J, 512]
        
        # Heatmap Generation
        hm_feat = self.heatmap_fc(feat)
        hm_feat = hm_feat.view(-1, self.base_channels, self.init_h, self.init_w)
        mu = self.deconv(hm_feat) # [B*T*J, 1, H, W]
        
        # Uncertainty Generation for Joint
        log_var = self.uncertainty_fc(feat) # [B*T*J, 1]
        
        mu = mu.view(B, T, J, self.H, self.W)
        log_var = log_var.view(B, T, J) # [B, T, J]
        
        return mu, log_var


# === Brancher ===
class Brancher(nn.Module):
    def __init__(self, device, num_joints, num_bones, n_layers, dim_in, dim_feat, dim_rep, dim_out,
                 mlp_ratio, act_layer, attn_drop, drop, drop_path,
                 use_layer_scale, layer_scale_init_value, use_adaptive_fusion,
                 num_heads, qkv_bias, qkv_scale, hierarchical,
                 use_temporal_similarity, temporal_connection_len,
                 use_tcn, graph_only, neighbour_num,
                 n_frames, frame_size, inference):
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
        self.inference = inference
        if inference:
            self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)

        act_mapper = {
            "gelu": nn.GELU,
            "relu": nn.ReLU
        }

        self.ag = MotionAGFormer(n_layers=n_layers,
                               dim_in=dim_in,
                               dim_feat=dim_feat,
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
        self.uncertainty_head = UncertaintyHead(dim_feat=dim_feat, dim_rep=dim_rep).to(device)


    def forward(self, x):
        # --- Estrai keypoints 2D in batch ---
        if self.inference:
            kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)
        else:
            kpt_tensor = x
        
        # --- Forward through AGFormer ---
        latent = self.ag(kpt_tensor)  # (B, T, J, dim_feat)

        # --- Pose 3D ---
        out3d = self.head3d(latent)  # (B, T, J, 3)

        # --- Rotations ---
        rotations = self.rotation_head(latent)  # (B, T, bones, 3, 3)

        # --- Heatmap ---
        uncertainty, log_var_hm = self.uncertainty_head(latent)  # (B, T, J, H, W), (B, T, J)
            
        return out3d, rotations, uncertainty, log_var_hm


# ---------------------------
# Factory
# ---------------------------
def create_model(inference=False, frame_size=(352, 640)):
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
        frame_size=frame_size,  #(H, W)
        inference=inference,
        device=cfg.device
    ).to(cfg.device)

    return model