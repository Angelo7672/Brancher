# lift3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ri-uso PositionalEncoding e TemporalTransformer dal tuo file (incollarli o importarli)
# Qui ne metto versioni compatte (assumendo che tu abbia già la versione dettagliata).
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


class Kpt2DEncoder(nn.Module):
    """
    Encoder che prende keypoints 2D (B, T, J, 2) e produce embedding (B, T, d_model).
    Implementazione semplice: flatten per frame e linear projection.
    """
    def __init__(self, num_joints: int, d_model: int, drop: float = 0.0):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = num_joints * 2
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        # x: (B, T, J, 2)
        B, T, J, C = x.shape
        assert J == self.num_joints and C == 2
        flat = x.view(B, T, J * 2)
        emb = self.fc(flat)  # (B, T, d_model)
        return emb


class Lift3DModel(nn.Module):
    """
    Pipeline: 2D keypoints -> encoder -> transformer -> 3D pose head
    """
    def __init__(self, num_joints: int, d_model: int = 512, depth: int = 6, heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.encoder_2d = Kpt2DEncoder(num_joints, d_model, drop)
        self.temporal = TemporalTransformer(d_model, depth, heads, mlp_ratio, drop)
        self.head = PoseHead(d_model, num_joints, drop)

    def forward(self, keypoints_2d):
        # keypoints_2d: (B, T, J, 2) normalized (es. in [0,1] o centered)
        emb = self.encoder_2d(keypoints_2d)  # (B, T, D)
        tokens = self.temporal(emb)          # (B, T, D)
        pose3d = self.head(tokens)           # (B, T, J, 3)
        # Root centering: substract pelvis (assumed joint 0)
        root = pose3d[:, :, :1, :]
        pose3d = pose3d - root
        return pose3d


# ---------- Loss utility (MPJPE) ----------
def mpjpe_loss(predicted, target, mask=None):
    """
    Mean per-joint position error (L2). predicted/target: (B, T, J, 3)
    optional mask: (B, T, J) with 1 for valid joints
    """
    diff = predicted - target
    error = torch.norm(diff, dim=-1)  # (B, T, J)
    if mask is not None:
        error = error * mask
        denom = mask.sum()
        return error.sum() / (denom + 1e-8)
    else:
        return error.mean()


# ---------- Inference pipeline helper ----------
def normalize_keypoints(keypoints, crop_bbox, img_size):
    """
    keypoints: list of (x,y,conf) of length J (local to crop)
    crop_bbox: (x1,y1,x2,y2) crop origin in full image
    img_size: (H, W) original image size
    Returns normalized keypoints in [0,1] relative to full image size -> tensor (J,2)
    """
    x1, y1, x2, y2 = crop_bbox
    H, W = img_size
    kp2 = []
    for kp in keypoints:
        if kp is None:
            kp2.append([0.0, 0.0])  # missing -> zero (puoi anche propagarla dall'ultimo frame)
        else:
            x_loc, y_loc, conf = kp
            x_full = x_loc + x1
            y_full = y_loc + y1
            # normalize
            kp2.append([x_full / float(W), y_full / float(H)])
    return np.array(kp2, dtype=np.float32)  # (J,2)


def run_inference_on_clip(frames, person_detector, hr_model, lift_model, device='cpu',
                          conf_thresh=0.2, clip_len=27, img_size=None):
    """
    frames: list/array of video frames (BGR numpy arrays), length >= clip_len
    person_detector: your YOLO model
    hr_model: HR wrapper (pretrained hrnet) put in eval mode
    lift_model: instance of Lift3DModel (eval)
    Returns: pose3d array (T, J, 3) for the selected person (first detected person)
    Nota: implementazione minimale per demo single-person single-clip.
    """
    lift_model.to(device)
    hr_model.to(device)
    hr_model.eval()
    lift_model.eval()

    # take first clip_len frames
    frames_clip = frames[:clip_len]
    J = lift_model.head.num_joints

    kps_seq = []  # list of (J,2)
    H = frames_clip[0].shape[0]
    W = frames_clip[0].shape[1]

    for f in frames_clip:
        # detect people
        results = person_detector(f)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) and results[0].boxes is not None else np.array([])
        cls = results[0].boxes.cls.cpu().numpy() if len(results) and results[0].boxes is not None else np.array([])

        # pick first person
        person_box = None
        for i, c in enumerate(cls):
            if int(c) == 0:
                person_box = boxes[i].astype(int)
                break
        if person_box is None:
            # no person -> append zeros
            kps_seq.append(np.zeros((J, 2), dtype=np.float32))
            continue

        x1, y1, x2, y2 = map(int, person_box)
        person_crop = f[y1:y2, x1:x2]
        if person_crop.size == 0:
            kps_seq.append(np.zeros((J, 2), dtype=np.float32))
            continue

        # HRNet preprocessing: reuse your preprocess(person_crop)
        # Assuming you have function preprocess() that yields tensor (1,3,256,192)
        input_tensor = preprocess(person_crop).to(device)
        with torch.no_grad():
            out = hr_model(input_tensor)  # heatmaps
            heatmaps = out[0].detach().cpu().numpy()  # (J, Hh, Ww)
            keypoints_local = extract_keypoints_with_confidence(heatmaps, orig_size=person_crop.shape[:2], conf_thresh=conf_thresh)
            keypoints_global = [(pt[0] + x1, pt[1] + y1, pt[2]) if pt is not None else None for pt in keypoints_local]
            # normalize to [0,1] full-image coords
            norm = normalize_keypoints(keypoints_local, (x1, y1, x2, y2), img_size=(H, W))
            kps_seq.append(norm)

    # shape to tensor (1, T, J, 2)
    kps_seq = np.stack(kps_seq, axis=0)  # (T, J, 2)
    kps_seq = np.transpose(kps_seq, (0, 1, 2))  # already T,J,2
    kps_tensor = torch.from_numpy(kps_seq).unsqueeze(0).float().to(device)  # (1, T, J, 2)

    with torch.no_grad():
        pred3d = lift_model(kps_tensor)  # (1, T, J, 3)
    pred3d = pred3d.squeeze(0).cpu().numpy()  # (T, J, 3)
    return pred3d
