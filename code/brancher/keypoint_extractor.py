from ultralytics import YOLO
from config import Config
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from mmpose.apis import init_model


# === HRNet wrapper ===
class HR(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=True):
        super(HR, self).__init__()

        self.model = init_model(config_file, checkpoint_file, device=device)

        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        """
        x: Tensor di forma [B, 3, H, W]
        Output: heatmap [B, num_joints, H_out, W_out]
        """
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        x = self.deconv_layer(features)
        x = self.final_layer(x)
        return x

# === ViTPose wrapper ===
class ViTPoseBase(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=True):
        super(ViTPoseBase, self).__init__()
        
        self.model = init_model(config_file, checkpoint_file, device=device)
        
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        """
        x: Tensor di forma [B, 3, H, W]
        Output: heatmap [B, num_joints, H_out, W_out]
        """
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        x = self.deconv_layer(features)
        x = self.final_layer(x)
        return x

# === Trasformazione input ===
def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.
    tensor = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(tensor)
    return tensor.unsqueeze(0)


# === Estrazione keypoint da heatmap ===
def extract_keypoints(heatmaps, orig_size=(0, 0), conf_thresh=0.35):
    num_joints, h, w = heatmaps.shape
    scale_x = orig_size[1] / w
    scale_y = orig_size[0] / h

    # Flatten
    heatmaps_flat = heatmaps.view(num_joints, -1)
    # Trova indice e confidenza massima per ciascun joint
    conf_max, idx_max = torch.max(heatmaps_flat, dim=1)

    # Converti indice lineare in coordinate (y, x)
    y = idx_max // w
    x = idx_max % w

    # Rescaling alle dimensioni originali
    x_orig = x.float() * scale_x
    y_orig = y.float() * scale_y

    keypoints = torch.stack([x_orig, y_orig, conf_max], dim=1)
    # Applica soglia di confidenza
    keypoints[conf_max < conf_thresh] = torch.nan

    return keypoints  # tensor (J,3)


# === Keypoint Extractor ===
class KeypointExtractor:
    def __init__(self):
        cfg = Config()

        #config_file = 'AR17/code/keypoints_module/hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'   
        #checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        config_file = 'AR17/code/keypoints_module/vitpose/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
        checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth'

        self.device = torch.device(cfg.device)
        #self.model = HR(config_file, checkpoint_file, device=self.device).to(self.device)
        self.model = ViTPoseBase(config_file, checkpoint_file, device=self.device).to(self.device)
        self.model.eval()
        self.person_detector = YOLO("./yolov8n.pt")
        self.model_h, self.model_w = cfg.frame_size  # (H, W)

    # === Step 1: Rileva persone in batch ===
    def detect_persons_batch(self, frames):
        results = self.person_detector(frames, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            return []

        all_boxes = []
        for res in results:
            boxes = []
            if res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                for i, box in enumerate(xyxy):
                    if int(cls[i]) == 0:  # solo persone
                        boxes.append(box)
            all_boxes.append(boxes)

        return all_boxes  # lista lunga T, ogni elemento = [[x1,y1,x2,y2], ...]

    # === Step 2: Estrai keypoints da più box ===
    def extract_from_boxes_batch(self, frames, all_boxes):
        """
        frames: lista di frame (T,H,W,3)
        all_boxes: lista lunga T, ogni elemento = [[x1,y1,x2,y2], ...]
        Ritorna: lista lunga T
        """
        crops = []
        crop_meta = []  # (frame_idx, x1, y1)

        # 1️⃣ Crea batch di crop
        for i, (frame, boxes) in enumerate(zip(frames, all_boxes)):
            H, W = frame.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                resized = cv2.resize(crop, (self.model_w, self.model_h))
                crops.append(preprocess(resized))
                crop_meta.append((i, x1, y1))

        if not crops:
            return [[] for _ in frames]

        # 2️⃣ Batch inference con ViTPose
        batch_tensor = torch.cat(crops, dim=0).to(self.device)  # [N,3,H,W]
        with torch.no_grad():
            outputs = self.model(batch_tensor)  # [N, num_joints, H_out, W_out]

        # 3️⃣ Estrazione keypoints e ricostruzione globale
        T = len(frames)
        all_keypoints = [[] for _ in range(T)]

        for (frame_idx, x1, y1), heatmap in zip(crop_meta, outputs):
            kps_local = extract_keypoints(heatmap, orig_size=(self.model_h, self.model_w))
            kps_local = kps_local.cpu().numpy()

            valid_count = np.sum(~np.isnan(kps_local[:, :2]).any(axis=1))
            if valid_count < 9: # scarta se meno di 9 keypoints validi
                continue

            keypoints_global = []
            for pt in kps_local:
                if np.isnan(pt).any():
                    keypoints_global.append(None)
                else:
                    keypoints_global.append((pt[0] + x1, pt[1] + y1, pt[2]))
            all_keypoints[frame_idx].append(keypoints_global)

        return all_keypoints


# === Keypoint Normalizer ===
class KeypointNormalizer:
    """
    Normalizza i keypoints 2D in coordinate [0,1] rispetto alla dimensione dell'immagine
    """
    def __init__(self, frame_size):
        self.height, self.width = frame_size

    def normalize(self, keypoints):
        normed = []
        for pt in keypoints:
            if pt is not None:
                x, y, conf = pt
                normed.append([x / self.width, y / self.height, conf])
            else:
                normed.append([-1.0, -1.0, 0.0])
        return np.array(normed, dtype=np.float32)


# === Keypoint Extractor per sequenze ===
class KeypointExtractorSeq:
    def __init__(self, num_joints: int = 17, frame_size=(256, 192)):
        """
        extractor: KeypointExtractor (con YOLO + ViTPose/HRNet)
        normalizer: KeypointNormalizer
        """
        self.extractor = KeypointExtractor()
        self.normalizer = KeypointNormalizer(frame_size=frame_size)
        self.num_joints = num_joints


    def __call__(self, x):
        """
        x: (B,T,H,W,3)
        return: np.array (B,T,J,3)
        """
        x_np = np.array(x)
        B, T = x_np.shape[0], x_np.shape[1]
        keypoints_out = np.zeros((B, T, self.num_joints, 3), dtype=np.float32)
        empty_kp = np.full((self.num_joints, 3), -1.0, dtype=np.float32)
        empty_kp[:, 2] = 0.0

        # === Prepara frames ===
        for b in range(B):
            frames_list = []
            for t in range(T):
                frame = x_np[b, t]
                frame = frame.astype(np.uint8)
                frames_list.append(frame)

            # === Step 1: YOLO batch detection ===
            boxes_seq = self.extractor.detect_persons_batch(frames_list)

            # === Step 2: Batch keypoint extraction ===
            all_keypoints_seq = self.extractor.extract_from_boxes_batch(frames_list, boxes_seq)

            for t in range(T):
                persons_kps = all_keypoints_seq[t]
                if len(persons_kps) == 0:
                    keypoints_out[b, t] = empty_kp
                    continue

                # Prendiamo solo la prima persona (single pose)
                kp_first = persons_kps[0]
                # pad/truncate
                kp_first = kp_first[:self.num_joints] + [None] * (self.num_joints - len(kp_first))
                keypoints_out[b, t] = self.normalizer.normalize(kp_first)

        return keypoints_out  # (B,T,J,3)