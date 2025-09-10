from ultralytics import YOLO
import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from collections import deque
from keypoint_extractor import preprocess, extract_keypoints_with_confidence, HR
from config import Config
import random


# === Utilities ===
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


# === Keypoint Extractor ===
class KeypointExtractor:
    def __init__(self):
        config_file = 'AR17/code/keypoints_module/hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'   
        checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = HR(config_file, checkpoint_file, device=self.device).to(self.device)
        self.model.eval()
        self.person_detector = YOLO("./yolov8n.pt")

    def extract(self, frame):
        # === Step 1: Detect people ===
        results = self.person_detector(frame)
        # gestisce eventualità in cui non ci siano boxes
        if len(results) == 0 or results[0].boxes is None:
            return []
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()

        keypoints_all = []

        # === Step 2: For each person box, extract keypoints ===
        for i, box in enumerate(boxes):
            if int(cls[i]) != 0:  # COCO class 0 == person
                continue
            x1, y1, x2, y2 = map(int, box)
            # attenzione ai limiti di crop
            H, W = frame.shape[:2]
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            input_tensor = preprocess(person_crop).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                heatmaps = output[0].detach().cpu().numpy()
                keypoints_local = extract_keypoints_with_confidence(heatmaps, orig_size=person_crop.shape[:2])
                # keypoints in full image coords
                keypoints_global = [(pt[0] + x1, pt[1] + y1) if pt is not None else None for pt in keypoints_local]
                keypoints_all.append(keypoints_global)

        return keypoints_all


# === Keypoint Normalizer ===
class KeypointNormalizer:
    def __init__(self, frame_size):
        """
        image_size: (H, W) dell'immagine di riferimento
        """
        self.height, self.width = frame_size

    def normalize(self, keypoints):
        """
        keypoints: lista di (x,y) oppure None
        ritorna: np.array shape (J,2) con coordinate normalizzate [0,1]
        """
        normed = []
        for pt in keypoints:
            if pt is None:
                normed.append([0.0, 0.0])  # keypoint mancante
            else:
                x, y = pt
                normed.append([x / self.width, y / self.height])
        return np.array(normed, dtype=np.float32)


# ==== Embedding 2D keypoints ===
class Kpt2DEncoder(nn.Module):
    def __init__(self, num_joints, in_dim=2, embed_dim=512, drop: float = 0.0):
        """
        num_joints: numero di giunti (es. COCO = 17)
        in_dim: dimensione input (x,y) → 2
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
        x: (B, T, J, 2) keypoints normalizzati
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
    def __init__(self, num_joints: int, d_model: int, depth: int, heads: int,
                 mlp_ratio: float, drop: float, frame_size=(1200, 1920)):
        """
        frame_size: (H, W) usata per il normalizer; passare la risoluzione dei frames in input
        """
        super().__init__()
        self.num_joints = num_joints
        self.keypoint_extractor = KeypointExtractor()
        self.normalizer = KeypointNormalizer(frame_size)
        self.encoder2d = Kpt2DEncoder(num_joints=num_joints,
                                      in_dim=2,
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
        x: può essere:
          - torch.Tensor shape (B, T, 3, H, W)  (canale-first) OR
          - torch.Tensor shape (B, T, H, W, 3)  (canale-last) OR
          - numpy array or list: (T, H, W, 3)  OR (B, T, H, W, 3)
        Nota: la funzione prende i frame, chiama KeypointExtractor.extract(frame) (che usa YOLO+HRNet),
              normalizza i keypoints e costruisce il tensore (B, T, J, 2) che viene poi passato al modello.
        Return: tensor (B, T, J, 3) on the same device of this module.
        """

        # --- 1) normalize input representation -> list of videos, each is list of frames (H,W,3 BGR numpy) ---
        videos = []  # list of videos; each video is a list of frames (numpy H,W,3)  DA CONTROLLARE QUANDO DOWNLOAD E" FINITO
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            if x_np.ndim == 5:
                # assume (B, T, 3, H, W) or (B, T, H, W, 3)
                B, T = x_np.shape[0], x_np.shape[1]
                if x_np.shape[2] == 3:
                    # (B, T, 3, H, W) -> convert to (H,W,3)
                    for b in range(B):
                        frames_b = []
                        for t in range(T):
                            frame = np.transpose(x_np[b, t], (1, 2, 0))  # H,W,3
                            frames_b.append(frame.copy())
                        videos.append(frames_b)
                else:
                    # (B, T, H, W, 3)
                    for b in range(B):
                        frames_b = [x_np[b, t].copy() for t in range(T)]
                        videos.append(frames_b)
            elif x_np.ndim == 4:
                # assume (T, H, W, 3)
                T = x_np.shape[0]
                videos.append([x_np[t].copy() for t in range(T)])
            else:
                raise ValueError("Unsupported torch tensor shape for forward()")
        elif isinstance(x, (list, np.ndarray)):
            arr = np.array(x)
            if arr.ndim == 4:
                # (T, H, W, 3)
                T = arr.shape[0]
                videos.append([arr[t].copy() for t in range(T)])
            elif arr.ndim == 5:
                # (B, T, H, W, 3)
                B = arr.shape[0]
                for b in range(B):
                    videos.append([arr[b, t].copy() for t in range(arr.shape[1])])
            else:
                raise ValueError("Unsupported ndarray/list shape for forward()")
        else:
            raise ValueError("Unsupported input type for forward()")

        # --- 2) per video, per frame: extract keypoints, normalize, build array (T, J, 2) ---
        batch_keypoints = []
        for vid in videos:
            per_frame_kps = []
            for frame in vid:
                # ensure uint8 BGR
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                # Keypoint extraction (list of persons)
                persons_kps = self.keypoint_extractor.extract(frame)
                # choose first person if present, else fill zeros
                if len(persons_kps) == 0:
                    kp_norm = np.zeros((self.num_joints, 2), dtype=np.float32)
                else:
                    kp_first = persons_kps[0]  # list of J (x,y) or None
                    # if kp_first has different length, handle by trunc/pad
                    if len(kp_first) != self.num_joints:
                        # safe pad/truncate
                        kp_fixed = []
                        for i in range(self.num_joints):
                            if i < len(kp_first):
                                kp_fixed.append(kp_first[i])
                            else:
                                kp_fixed.append(None)
                        kp_first = kp_fixed
                    kp_norm = self.normalizer.normalize(kp_first)  # (J,2)
                per_frame_kps.append(kp_norm)
            # stack frames -> (T, J, 2)
            seq = np.stack(per_frame_kps, axis=0)
            batch_keypoints.append(seq)

        # --- 3) to torch, shape (B, T, J, 2) ---
        batch_keypoints = np.stack(batch_keypoints, axis=0)  # (B, T, J, 2)
        device = next(self.parameters()).device
        kpt_tensor = torch.from_numpy(batch_keypoints).float().to(device)  # (B, T, J, 2)

        # --- 4) forward through encoder / transformer / head ---
        enc = self.encoder2d(kpt_tensor)   # (B, T, D)
        temp = self.temporal(enc)          # (B, T, D)
        out3d = self.head3d(temp)          # (B, T, J, 3)

        return out3d


# ---------------------------
# Factory
# ---------------------------
def create_model():
    cfg = Config()
    set_seed(cfg.seed)

    model = Lift3DModel(
        num_joints=cfg.num_joints,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
        frame_size=(cfg.img_size, cfg.img_size)  # se non square, cambia con (H,W)
    ).to(cfg.device)

    print(f"Model params: {count_params(model)/1e6:.2f}M")

    summary(
        model,
        input_size=(1, cfg.clip_len, cfg.num_joints, 2),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        device=cfg.device,
    )

    return model