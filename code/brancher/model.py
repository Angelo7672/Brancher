import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from collections import deque
from config import Config
from DSTformer import DSTformer
import random
import cv2


# === Utilities ===
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# === Lift3DModel ===
class Brancher(nn.Module):
    """
    Usa DSTformer che prende direttamente keypoints 2D normalizzati (B,T,J,2)
    e restituisce 3D (B,T,J,3).
    """
    def __init__(self, num_joints: int, d_model: int, depth: int, heads: int,
                 mlp_ratio: float, drop: float, frame_size=(256, 192),
                 maxlen: int = 243, att_fuse: bool = True):
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
        self.keypoint_extractor = KeypointExtractor(frame_size)
        self.normalizer = KeypointNormalizer(frame_size)

        # DSTformer si aspetta input (B,T,J,C). Qui C=2 (x,y) normalizzati.
        # impostiamo dim_in=2 e dim_feat=d_model; dim_out=3 per ottenere 3D coords
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
        ).to(next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device('cpu'))

        # Nota: DSTformer ha già la head che ritorna dim_out=3; non servono ulteriori MLP

    def forward(self, x):
        """
        x: torch.Tensor o np.array/list
           - torch.Tensor (B, T, H, W, 3)
           - np.array/list (B, T, H, W, 3)
        return: torch.Tensor (B, T, J, 3)
        """
        # --- 1) Convert input to numpy frames list ---
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)

        B, T = x_np.shape[0], x_np.shape[1]
        videos = []
        for b in range(B):
            frames_b = []
            for t in range(T):
                frame = x_np[b, t]
                if frame.shape[0] == 3:  # channel-first
                    frame = np.transpose(frame, (1, 2, 0))
                frames_b.append(frame.copy())
            videos.append(frames_b)

        # --- 2) Extract keypoints and normalize ---
        batch_keypoints = []
        for vid in videos:
            per_frame_kps = []
            for frame in vid:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                persons_kps = self.keypoint_extractor.extract(frame)
                if len(persons_kps) == 0:
                    kp_norm = np.zeros((self.num_joints, 3), dtype=np.float32)
                else:
                    kp_first = persons_kps[0]
                    # pad/truncate if needed
                    kp_first = kp_first[:self.num_joints] + [None] * (self.num_joints - len(kp_first))
                    kp_norm = self.normalizer.normalize(kp_first)
                per_frame_kps.append(kp_norm)
            batch_keypoints.append(np.stack(per_frame_kps, axis=0))  # (T, J, 3)

        # kpt_tensor: (B, T, J, 3)
        kpt_tensor = torch.from_numpy(np.stack(batch_keypoints, axis=0)).float().to(next(self.dst.parameters()).device)

        # --- 3) Forward through DSTformer ---
        # DSTformer.forward expects x shape [B, F, J, C]
        out3d = self.dst(kpt_tensor)  # (B, T, J, 3)

        return out3d


# ---------------------------
# Factory
# ---------------------------
def create_model():
    cfg = Config()
    set_seed(cfg.seed)

    model = Brancher(
        num_joints=cfg.num_joints,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
        frame_size=cfg.img_size  #(H, W)
    ).to(cfg.device)

    summary(
        model,
        input_size=(cfg.batch_size, cfg.clip_len, cfg.img_size[0], cfg.img_size[1], 3),    # (B, T, W, H, 3)
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        device=cfg.device,
    )

    return model