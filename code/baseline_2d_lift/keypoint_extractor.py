from ultralytics import YOLO
import ultralytics
#ultralytics.utils.LOGGER.setLevel("ERROR")
from config import Config
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
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

    # === Estrazione keypoint da heatmap ===
    def extract_keypoints(self, heatmaps, metas, device='cpu'):
        """
        Converte i keypoint dalle heatmap locali (crop) alle coordinate globali nel frame originale.

        Args:
            heatmaps (torch.Tensor): tensor [T, J, Hh, Wh] con le heatmap per ogni crop
            metas (list[tuple]): lista di tuple (frame_idx, x1, y1, crop_h, crop_w)
            device (str or torch.device): device di esecuzione (default: 'cuda')

        Returns:
            keypoints_global (torch.Tensor): tensor [T, J, 3] con (x_global, y_global, conf)
        """
        B, J, Hh, Wh = heatmaps.shape

        # === Flatten e trova il massimo per ogni joint ===
        heatmaps_flat = heatmaps.view(B, J, -1)
        conf_max, idx_max = torch.max(heatmaps_flat, dim=2)  # [T, J]

        # Coordinate locali (nella heatmap)
        y_local = (idx_max // Wh).float()
        x_local = (idx_max % Wh).float()

        # === Recupera i metadati ===
        metas_tensor = torch.tensor(metas, device=device, dtype=torch.float32)  # [T, 5]
        _, x1, y1, crop_h, crop_w = metas_tensor.T  # ognuno [T]

        # === Calcola scaling per ogni crop ===
        scale_x = crop_w / Wh
        scale_y = crop_h / Hh

        # === Riscalamento e traslazione alle coordinate globali ===
        x_global = x_local * scale_x[:, None] + x1[:, None]
        y_global = y_local * scale_y[:, None] + y1[:, None]

        # === Stack finale ===
        keypoints_global = torch.stack([x_global, y_global, conf_max], dim=2)  # [T, J, 3]

        return keypoints_global

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
            heatmaps = self.model(crop_batch)  # [T, J, H_out, W_out]

        keypoints = self.extract_keypoints(heatmaps, metas, device=self.device)  # [T, J, 3]

        # === Assegnazione finale ai frame ===
        for meta, kps in zip(metas, keypoints):
            frame_idx = int(meta[0])
            keypoints_global = [
                (float(px), float(py), float(cf)) for px, py, cf in kps
            ]
            all_keypoints[frame_idx].append(keypoints_global)

        return all_keypoints


# === Keypoint Normalizer ===
class KeypointNormalizer:
    """
    Normalizza i keypoints 2D in coordinate [0,1] rispetto alla dimensione dell'immagine
    """
    def __init__(self, frame_size, device):
        self.height = float(frame_size[0])
        self.width = float(frame_size[1])
        self.device = device

    def normalize(self, keypoints):
        # Pre-allocate output tensor
        num_kp = len(keypoints)
        normed = torch.empty((num_kp, 3), dtype=torch.float32, device=self.device)
        
        for i, pt in enumerate(keypoints):
            x, y, conf = pt
            normed[i] = torch.tensor([x / self.width, y / self.height, conf], device=self.device)
              
        return normed


# === Keypoint Extractor per sequenze ===
class KeypointExtractorSeq:
    def __init__(self, frame_size, device, num_joints: int = 17):
        """
        extractor: KeypointExtractor (con YOLO + Pose Extractor)
        normalizer: KeypointNormalizer
        """
        self.extractor = KeypointExtractor()
        self.normalizer = KeypointNormalizer(frame_size=frame_size, device=device)
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
            keypoints_all = self.extractor.extract_from_boxes(video_uint8, boxes_all)

            # === Step 3: Keypoint normalization and assignment ===
            for t, keypoints in enumerate(keypoints_all):
                if len(keypoints) == 0:
                    keypoints_out[b, t] = empty_kp
                else:
                    # Take first person's keypoints
                    kp_first = keypoints[0]
                    # pad/truncate
                    kp_first = kp_first[:self.num_joints] + [None] * (self.num_joints - len(kp_first))
                    keypoints_out[b, t] = self.normalizer.normalize(kp_first)

        return keypoints_out  # (B,T,J,3)