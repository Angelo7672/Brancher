from ultralytics import YOLO
import ultralytics
from config import Config
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.io import read_video
from mmpose.apis import init_model


# === 2D Pose Extractor ===
class PE(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=True):
        super(PE, self).__init__()

        self.model = init_model(config_file, checkpoint_file, device=device)

        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        """
        x: Tensor di forma [T, 3, H, W]
        Output: heatmap [T, num_joints, H_out, W_out]
        """
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        x = self.deconv_layer(features)
        x = self.final_layer(x)
        return x


# === Keypoint Extractor ===
class KeypointExtractor:
    def __init__(self):
        cfg = Config()

        self.device = torch.device(cfg.device)
        self.model = PE(cfg.config_file, cfg.checkpoint_file, device=self.device).to(self.device)
        self.model.eval()
        self.person_detector = YOLO(cfg.yolo_version).to(self.device)
        self.person_detector.model.to(self.device)
        self.person_detector.model.eval()  # ensure model is in eval mode
        self.model_h, self.model_w = cfg.pe_size  # (H, W)
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def map_heatmap_back(self, hmap, x1, y1, crop_h, crop_w, H, W):
        """
        hmap: heatmap (J, Hh, Wh) prodotta da PE sul crop
        x1,y1: posizione crop
        crop_h, crop_w: dimensioni crop originale
        H, W: dimensioni frame intero
        """
        J, Hh, Wh = hmap.shape

        # === Resize della heatmap alla dimensione del crop originale ===
        hmap_resized = torch.nn.functional.interpolate(
            hmap.unsqueeze(0), size=(crop_h, crop_w),
            mode="bilinear", align_corners=False
        )[0]   # → (J, crop_h, crop_w)

        # === Canvas vuoto della dimensione del frame ===
        hmap_full = torch.zeros((J, H, W), device=hmap.device, dtype=hmap.dtype)

        # === Incolla la heatmap nella posizione corretta ===
        hmap_full[:, y1:y1+crop_h, x1:x1+crop_w] = hmap_resized

        return hmap_full


    # === Step 1: Rileva persone in batch ===
    def detect_persons(self, frames):
        """
        frames: torch array (T,H,W,3) - batch of frames
        Returns: list of lists of boxes [[x1,y1,x2,y2], ...] for each frame
        """
        if len(frames) == 0:
            return []

        # Process batch tensor: permute to CHW and normalize to [0,1]
        batch = frames.permute(0, 3, 1, 2).float().to(self.device) / 255.0  # (T, 3, H, W)
        
        with torch.no_grad():
            # YOLO's high-level API with our preprocessed batch
            results = self.person_detector(batch, stream=False, verbose=False)

        # Process all results in a vectorized way
        all_boxes = []
        empty_tensor = torch.zeros((0, 4), device=self.device)  # Empty bbox tensor
        
        for result in results:
            if len(result.boxes) == 0:
                all_boxes.append(empty_tensor)
                continue
                
            # Filter person class (0) and get boxes
            mask = result.boxes.cls == 0
            if not mask.any():  # No person detections
                all_boxes.append(empty_tensor)
                continue
                
            # Get boxes and confidences
            person_boxes = result.boxes.xyxy[mask]
            confs = result.boxes.conf[mask]
            
            # Take the most confident detection
            _, idx = torch.topk(confs, k=1)  # Gets the index of highest confidence
            boxes = person_boxes[idx]  # Keep as torch tensor
            all_boxes.append(boxes)

        return all_boxes

    # === Step 2: Estrai keypoints da più box ===
    def extract_from_boxes(self, frames, all_boxes):
        """
        frames: torch.Tensor (T,H,W,3)
        all_boxes: list of length T, each element is a tensor of shape (1,4) containing [x1,y1,x2,y2]
        Returns: list of length T, each = list of keypoints_global
        """
        T = len(frames)
        all_keypoints = [[] for _ in range(T)]

        # === Quick return ===
        if all(b.size(0) == 0 for b in all_boxes):
            return all_keypoints

        crops, metas = [], []
        for t, (frame, boxes) in enumerate(zip(frames, all_boxes)):
            if boxes.size(0) == 0:
                continue

            H, W = frame.shape[:2]
            for box in boxes:
                # Clip box coordinates to image boundaries using torch operations
                x1, y1, x2, y2 = torch.clamp(box.to(torch.int32), 
                                           min=torch.tensor([0, 0, 0, 0], device=box.device),
                                           max=torch.tensor([W-1, H-1, W, H], device=box.device))
                
                if x2 <= x1 or y2 <= y1:
                    continue

                # Extract crop using tensor indexing
                crop = frame[y1:y2, x1:x2]
                if crop.numel() == 0:  # Check if crop is empty
                    continue

                # Permute and normalize to [0, 1]
                crop_tensor = crop.permute(2, 0, 1).float() / 255.
                crop_h, crop_w = crop.shape[:2]

                resized = torch.nn.functional.interpolate(crop_tensor.unsqueeze(0),
                                                          size=(self.model_h, self.model_w),
                                                          mode='bilinear', align_corners=False)[0]
                resized = resized.to(self.device)
                
                # Resize to exact model size
                crops.append(self.normalizer(resized))
                metas.append((t, x1, y1, crop_h, crop_w))  # store crop width and height for mapping

        if len(crops) == 0:
            return all_keypoints

        # Stack all preprocessed crops
        crop_batch = torch.stack(crops).to(self.device)

        # === Batch inference ===
        with torch.no_grad():
            heatmaps_batch = self.model(crop_batch)  # [T, J, H_out, W_out]
        
        heatmaps_out = [None] * T

        for i, (t, x1, y1, crop_h, crop_w) in enumerate(metas):
            frame = frames[t]
            H, W = frame.shape[0], frame.shape[1]
        
            heatmaps_out[t] = self.map_heatmap_back(
                heatmaps_batch[i], x1, y1, crop_h, crop_w, H, W
            )

        return heatmaps_out


# === Keypoint Extractor per sequenze ===
class KeypointExtractorSeq:
    def __init__(self, frame_size, device, num_joints: int = 17):
        """
        extractor: KeypointExtractor (con YOLO + Pose Extractor)
        """
        self.extractor = KeypointExtractor()
        self.num_joints = num_joints
        self.device = device

    def __call__(self, x):
        """
        x: torch.Tensor (B,T,H,W,3)
        return: torch.Tensor (B,T,J,3)
        """
        x_torch = x.to(self.device, non_blocking=True) 
        B, T = x_torch.shape[0], x_torch.shape[1]
        keypoints_out = torch.zeros((B, T, self.num_joints, 3), dtype=torch.float32, device=self.device)
        empty_kp = torch.full((self.num_joints, 3), -1.0, dtype=torch.float32, device=self.device)
        empty_kp[:, 2] = 0.0 # confidence zero for missing keypoints

        for b in range(B):
            video = x_torch[b]  # (T, H, W, 3)
            video_uint8 = video.to(torch.uint8).to(self.device)

            # === Step 1: YOLO detection ===
            boxes_all = self.extractor.detect_persons(video_uint8)

            # === Step 2: Keypoint extraction ===
            heatmaps = self.extractor.extract_from_boxes(video_uint8, boxes_all)

        return heatmaps


frame_size = (640, 416)  # Per YOLO
num_joints = 17
device = "cuda" if torch.cuda.is_available() else "cpu"
keypoint_extractor = KeypointExtractorSeq(num_joints=num_joints, frame_size=frame_size, device=device)
video_path = "/Experiments/angeloa00/SportsPose_reprocessed_v2/sample_00270_view_back/video.mp4"

# === Carica il video direttamente come tensore ===
video_frames, audio, info = read_video(video_path, output_format="THWC")

# video_frames: (T, H, W, 3)

# === Aggiungi batch dimension ===
video_tensor = video_frames.unsqueeze(0)     # (1, T, H, W, 3)

print("Shape video:", video_tensor.shape)

heatmap = keypoint_extractor(video_tensor)
# Filtra i None
valid = [h for h in heatmap if h is not None]

if len(valid) == 0:
    print("Nessuna heatmap trovata")
else:
    h = valid[0]
    T = len(heatmap)
    heatmaps = torch.zeros((1, T, h.shape[0], h.shape[1], h.shape[2]), dtype=torch.float16)
    print("Heatmap totale shape =", heatmaps.shape)

torch.save(heatmaps, "heatmaps.pt")
