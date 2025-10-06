from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from mmpose.apis import init_model
from PIL import Image


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
def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


# === Estrazione keypoint da heatmap ===
def extract_keypoints(heatmaps, input_size=(256, 192), orig_size=(0, 0), conf_thresh=0.4):
    input_h, input_w = input_size
    heatmap_h, heatmap_w = heatmaps.shape[1:]
    scale_x = (orig_size[1] / heatmap_w)
    scale_y = (orig_size[0] / heatmap_h)

    keypoints = []
    for heatmap in heatmaps:
        conf = np.max(heatmap)
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # --- refinement sub-pixel ---
        if 1 < x < heatmap_w-1 and 1 < y < heatmap_h-1:
            diff_x = heatmap[y, x+1] - heatmap[y, x-1]
            diff_y = heatmap[y+1, x] - heatmap[y-1, x]
            x += np.sign(diff_x) * 0.25
            y += np.sign(diff_y) * 0.25

        x_orig = x * scale_x
        y_orig = y * scale_y

        if conf >= conf_thresh:
            keypoints.append((float(x_orig), float(y_orig), float(conf)))
        else:
            keypoints.append(None)

    return keypoints


# === Keypoint Extractor ===
class KeypointExtractor:
    def __init__(self, frame_size):
        #config_file = 'AR17/code/keypoints_module/hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'   
        #checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        config_file = 'AR17/code/keypoints_module/vitpose/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
        checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.model = HR(config_file, checkpoint_file, device=self.device).to(self.device)
        self.model = ViTPoseBase(config_file, checkpoint_file, device=self.device).to(self.device)
        self.model.eval()
        self.person_detector = YOLO("./yolov8n.pt")
        self.model_h, self.model_w = frame_size  # (H, W)

    def extract(self, frame):
        # === Step 1: Detect people ===
        results = self.person_detector(frame, verbose=False)
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

            # crop + resize
            H, W = frame.shape[:2]
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            person_resized = cv2.resize(person_crop, (self.model_w, self.model_h))  # resize a 256x192

            input_tensor = preprocess(person_resized).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                heatmaps = output[0].detach().cpu().numpy()
                keypoints_local = extract_keypoints(heatmaps, orig_size=(self.model_h, self.model_w))
                valid_count = sum(1 for pt in keypoints_local if pt is not None)
                if valid_count < 9 :  # Require at least 9 valid keypoints
                    continue

                keypoints_global = [
                    (pt[0] + x1, pt[1] + y1, pt[2]) if pt is not None else None
                    for pt in keypoints_local
                ]

                keypoints_all.append(keypoints_global)

        return keypoints_all


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
        self.extractor = KeypointExtractor(frame_size=frame_size)
        self.normalizer = KeypointNormalizer(frame_size=frame_size)
        self.num_joints = num_joints

    def __call__(self, x):
        """
        x: np.array o torch.Tensor con shape (B,T,H,W,3)
        return: np.array (B,T,J,3)
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)

        B, T = x_np.shape[0], x_np.shape[1]
        batch_keypoints = []

        for b in range(B):
            per_frame_kps = []
            for t in range(T):
                frame = x_np[b, t]
                if frame.shape[0] == 3:  # channel-first
                    frame = np.transpose(frame, (1, 2, 0))

                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                persons_kps = self.extractor.extract(frame)
                if len(persons_kps) == 0:
                    kp_norm = np.array([[-1.0, -1.0, 0.0]] * self.num_joints, dtype=np.float32)
                else:
                    kp_first = persons_kps[0]
                    # pad/truncate
                    kp_first = kp_first[:self.num_joints] + [None] * (self.num_joints - len(kp_first))
                    kp_norm = self.normalizer.normalize(kp_first)

                per_frame_kps.append(kp_norm)
            batch_keypoints.append(np.stack(per_frame_kps, axis=0))  # (T,J,3)

        return np.stack(batch_keypoints, axis=0)  # (B,T,J,3)