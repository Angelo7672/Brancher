import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from collections import deque
from config import Config
from DSTformer import DSTformer
from keypoint_extractor import KeypointExtractorSeq
import random
import cv2
from scipy.interpolate import interp1d


# === Utilities ===
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def interpolate_missing_joints(keypoints_2d):
    """
    Interpola (x,y) per i joint con conf=0 lungo la dimensione temporale.
    keypoints_2d: np.array (B, T, J, 3)
    Ritorna: np.array (B, T, J, 3) con valori interpolati e conf=0.35 per i punti stimati.
    """
    B, T, J, _ = keypoints_2d.shape
    keypoints_interp = keypoints_2d.copy()

    for b in range(B):
        for j in range(J):
            x = keypoints_2d[b, :, j, 0]
            y = keypoints_2d[b, :, j, 1]
            c = keypoints_2d[b, :, j, 2]

            valid_idx = np.where(c > 0)[0]

            if len(valid_idx) == 0: # nessun punto valido
                continue  
            if len(valid_idx) == 1: # un solo punto valido
                keypoints_interp[b, :, j, 0] = x[valid_idx[0]]
                keypoints_interp[b, :, j, 1] = y[valid_idx[0]]
                keypoints_interp[b, :, j, 2] = 0.35  # tutte stimate
                keypoints_interp[b, valid_idx[0], j, 2] = c[valid_idx[0]]  # punto originale mantiene conf
                continue

            # Crea interpolatori per x e y
            f_x = interp1d(valid_idx, x[valid_idx], kind='linear', fill_value='extrapolate')
            f_y = interp1d(valid_idx, y[valid_idx], kind='linear', fill_value='extrapolate')

            all_idx = np.arange(T)
            x_interp = f_x(all_idx)
            y_interp = f_y(all_idx)

            # Aggiorna solo i frame mancanti
            missing_idx = np.where(c == 0)[0]
            keypoints_interp[b, missing_idx, j, 0] = x_interp[missing_idx]
            keypoints_interp[b, missing_idx, j, 1] = y_interp[missing_idx]
            keypoints_interp[b, missing_idx, j, 2] = 0.35  # conf 0.35 per punti stimati

    return keypoints_interp



# === BrancherModel ===
class Brancher(nn.Module):
    """
    Usa DSTformer che prende direttamente keypoints 2D normalizzati (B,T,J,2)
    e restituisce 3D (B,T,J,3).
    """
    def __init__(self, device, num_joints: int, d_model: int, depth: int, heads: int,
                 mlp_ratio: float, drop: float, frame_size: int,
                 maxlen: int = 243, att_fuse: bool = True, inference: bool = False):
        """
        num_joints: numero di giunti
        d_model: dimensione feature (dim_feat per DSTformer)
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
        self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size)
        self.inference = inference
        self.device = torch.device(device)

        self.dst = DSTformer(
            dim_in=3,           # input per joint: (x,y,conf)
            dim_out=3,          # output per joint: (x,y,z)
            dim_feat=d_model,   # feature intermedia
            dim_rep=d_model,    # rappresentazione prima della testa
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
        ).to(self.device)


    def forward(self, x):
        """
        x: (B,T,H,W,3).
        Output: torch.Tensor (B,T,J,3)
        """
        # --- 1️⃣ Estrai keypoints 2D in batch ---
        keypoints_2d = self.keypoint_extractor(x)  # np.array (B,T,J,3)

        # Interpola solo in inferenza (per ora)
        if self.inference:
            keypoints_2d = interpolate_missing_joints(keypoints_2d)

        # --- 2️⃣ Conversione in tensore ---
        kpt_tensor = torch.from_numpy(keypoints_2d).float().to(self.device)  # (B,T,J,3)

        # TODO: augmentation 2D

        # --- 3️⃣ Forward nel DSTformer ---
        out3d = self.dst(kpt_tensor)  # (B,T,J,3)

        return out3d


# ---------------------------
# Factory
# ---------------------------
def create_model(inference: bool = False):
    cfg = Config()
    set_seed(cfg.seed)

    model = Brancher(
        num_joints=cfg.num_joints,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
        frame_size=cfg.frame_size,
        inference=inference,
        device=cfg.device
    )

    summary(
        model,
        input_size=(cfg.batch_size, cfg.clip_len, cfg.frame_size[0], cfg.frame_size[1], 3),    # (B, T, W, H, 3)
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        device=cfg.device,
    )

    return model