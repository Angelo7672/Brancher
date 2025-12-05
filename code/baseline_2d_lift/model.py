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

class PositionalEncoding(nn.Module):    # SOLO TEMPORAL,  NON SPATIAL
    """
    Original PE from Attention is All You Need
    """
    def __init__(self, d_model, max_seq_len=243):
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

def extract_keypoints_from_heatmap(hmap, frame_size):
    """
    hmap: [B, T, J, H, W]
    returns: [B, T, J, 3] -> (x, y, conf)
    """
    B, T, J, H, W = hmap.shape
    keypoints = torch.zeros((B, T, J, 3), dtype=torch.float32, device=hmap.device)

    for b in range(B):
        for t in range(T):
            for j in range(J):
                h = hmap[b, t, j]  # [H, W]
                conf, idx = torch.max(h.view(-1), 0)
                y, x = divmod(int(idx.item()), W)
                keypoints[b, t, j] = torch.tensor([x, y, conf], device=hmap.device)
    
    # Normalize once after extraction
    keypoints[..., 0] = (keypoints[..., 0] / frame_size[1] - 0.5) * 2  # x: [0, W] -> [-1, 1]
    keypoints[..., 1] = (keypoints[..., 1] / frame_size[0] - 0.5) * 2  # y: [0, H] -> [-1, 1]
    
    return keypoints


# === Trasformer temporale ===
class TemporalTransformer(nn.Module):
    """
    Questa è la parte della rete che cattura relazioni temporali tra i frame della sequenza di pose.
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
    def __init__(self, device, num_joints: int, d_model: int, depth: int, heads: int, frame_size=(640, 416)):
        """
        frame_size: (H, W) usata per il normalizer; passare la risoluzione dei frames in input
        """
        super().__init__()
        self.num_joints = num_joints
        self.frame_size = frame_size
        self.keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)
        self.encoder2d = nn.Linear(num_joints * 3, d_model) # il PE LAVORA SULLA DIMENSIONE T, I TOKEN SONO I FRAME, NON I JOINT
        self.temporal = TemporalTransformer(d_model=d_model, depth=depth, nhead=heads).to(device)
        self.head3d = PoseHead(dim_feat=d_model, dim_rep=d_model, dim_out=num_joints*3).to(device)

    def forward(self, x):
        """
        x: torch.Tensor (B,T,H,W,3).
        Output: torch.Tensor (B,T,J,3)
        """
        # --- 1️⃣ Estrai keypoints 2D in batch ---
        kpt_tensor = self.keypoint_extractor(x)  # torch.Tensor (B,T,J,3)
        #kpt_tensor = extract_keypoints_from_heatmap(x, self.frame_size)  # torch.Tensor (B,T,J,3)
        #kpt_tensor = x  # Input già in forma di keypoints (B,T,J,3)

        # --- 2️⃣ Embedding ---
        B, T, J, D = kpt_tensor.shape
        kpt_flat = kpt_tensor.reshape(B, T, J * D)  # (B, T, J*3)
        enc = self.encoder2d(kpt_flat) # (B,T,embed_dim)

        # --- 3️⃣ Temporal Transformer ---
        tmp = self.temporal(enc)  # (B,T,embed_dim)

        # --- 4️⃣ Regressione 3D ---
        out3d = self.head3d(tmp)  
        out3d = out3d.reshape(B, T, self.num_joints, 3) # (B,T,J,3)

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
        frame_size=cfg.frame_size  #(H, W)
    ).to(cfg.device)

    #summary(
    #    model,
    #    input_size=(cfg.batch_size, cfg.clip_len, cfg.frame_size[0], cfg.frame_size[1], 3),    # (B, T, W, H, 3)
    #    #input_size=(cfg.batch_size, cfg.clip_len, cfg.num_joints, 64, 48),
    #    #input_size=(cfg.batch_size, cfg.clip_len, cfg.num_joints, 3),
    #    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    #    depth=3,
    #    device=cfg.device,
    #)

    return model

if __name__ == "__main__":
    create_model()