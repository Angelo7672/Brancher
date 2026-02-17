from ultralytics import YOLO
import ultralytics
from config import Config
import torch
import torch.nn as nn
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

    def extract_keypoints(self, heatmaps, metas, device='cpu'):
        """
        Converts keypoints from local heatmaps (crops) to global coordinates in the original frame.
        """
        B, J, Hh, Wh = heatmaps.shape

        # === Flatten and find the maximum for each joint ===
        heatmaps_flat = heatmaps.view(B, J, -1)
        conf_max, idx_max = torch.max(heatmaps_flat, dim=2)  # [T, J]

        # === Local coordinates (in the heatmap) ===
        y_local = (idx_max // Wh).float()
        x_local = (idx_max % Wh).float()

        # === Retrieve metadata ===
        metas_tensor = torch.tensor(metas, device=device, dtype=torch.float32)  # [T, 5]
        _, x1, y1, crop_h, crop_w = metas_tensor.T

        # === Calculate scaling for each crop ===
        scale_x = crop_w / Wh
        scale_y = crop_h / Hh

        # === Scaling and translation to global coordinates ===
        x_global = x_local * scale_x[:, None] + x1[:, None]
        y_global = y_local * scale_y[:, None] + y1[:, None]

        # === Final stack ===
        keypoints_global = torch.stack([x_global, y_global, conf_max], dim=2)  # [T, J, 3]

        return keypoints_global

    def detect_persons(self, frames):
        """
        Detect persons bounding box in each frame
        """
        if len(frames) == 0:
            return []

        # Preprocessing batch
        batch = frames.permute(0, 3, 1, 2).float().to(self.device) / 255.0  # (T, 3, H, W)

        with torch.no_grad():
            results = self.person_detector(batch, stream=False, verbose=False)

        all_boxes = []

        # === Sanitization of the starting point ===
        last_valid_box = None
        for res in results:
            mask = res.boxes.cls == 0
            if mask.any():
                confs = res.boxes.conf[mask]
                _, idx = torch.topk(confs, k=1)
                last_valid_box = res.boxes.xyxy[mask][idx]
                break

        if last_valid_box is None:
            last_valid_box = torch.zeros((1, 4), device=self.device)

        # === Time-fill processing ===
        for result in results:
            mask = result.boxes.cls == 0

            if mask.any():
                # Present detention: take the most reliable one
                confs = result.boxes.conf[mask]
                _, idx = torch.topk(confs, k=1)
                current_box = result.boxes.xyxy[mask][idx]
                last_valid_box = current_box
            else:
                # Absent detention: use the last valid one (or the first sanitized one)
                current_box = last_valid_box

            all_boxes.append(current_box)

        return all_boxes

    def extract_from_boxes(self, frames, all_boxes):
        """
        Extract 2d keypoints and confidence from all the boxes of the sequence
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

        crop_batch = torch.stack(crops).to(self.device)

        # === Batch inference ===
        with torch.no_grad():
            heatmaps = self.model(crop_batch)  # [T, J, H_out, W_out]

        keypoints = self.extract_keypoints(heatmaps, metas, device=self.device)  # [T, J, 3]

        # === Final assignment to frames ===
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
        Normalizes 2D keypoints in camera-normalized space:
        x ∈ [-1, 1]
        y ∈ [-h/w, h/w]
    """
    def __init__(self, frame_size, device):
        # I assume that frame_size is (Height, Width) as per numpy/torch shape
        # If frame_size comes from OpenCV(W,H), swap the indices below
        self.height = float(frame_size[0])
        self.width = float(frame_size[1])
        self.aspect_ratio = self.height / self.width
        self.device = device

    def normalize(self, keypoints):
        keypoints = torch.tensor(keypoints, dtype=torch.float32, device=self.device)
        normed = keypoints.clone()

        normed[..., 0] = keypoints[..., 0] / self.width * 2 - 1
        normed[..., 1] = keypoints[..., 1] / self.width * 2 - self.aspect_ratio
        
        return normed


# === Keypoint Extractor for sequences ===
class KeypointExtractorSeq:
    def __init__(self, frame_size, device, num_joints: int = 17):
        """
        extractor: KeypointExtractor (with YOLO + Pose Extractor)
        normalizer: KeypointNormalizer
        """
        self.extractor = KeypointExtractor()
        self.normalizer = KeypointNormalizer(frame_size=frame_size, device=device)
        self.num_joints = num_joints
        self.device = device
        self.notation = 'H36M' # or 'COCO'
    
    def turn_into_h36m(self, keypoints_out):
        """
        Mapping from COCO (17 joints) to H36M (17 joints)
        """
        # Empty tensor [B, T, 17, 3]
        h36m_kps = torch.zeros_like(keypoints_out)

        # 0: Pelvis (Center between the hips)
        h36m_kps[..., 0, :] = (keypoints_out[..., 11, :] + keypoints_out[..., 12, :]) * 0.5
        # 1: Right Hip
        h36m_kps[..., 1, :] = keypoints_out[..., 12, :]
        # 2: Right Knee
        h36m_kps[..., 2, :] = keypoints_out[..., 14, :]
        # 3: Right Ankle
        h36m_kps[..., 3, :] = keypoints_out[..., 16, :]
        # 4: Left Hip
        h36m_kps[..., 4, :] = keypoints_out[..., 11, :]
        # 5: Left Knee
        h36m_kps[..., 5, :] = keypoints_out[..., 13, :]
        # 6: Left Ankle
        h36m_kps[..., 6, :] = keypoints_out[..., 15, :]
        # 8: Thorax (Center between the shoulders)
        h36m_kps[..., 8, :] = (keypoints_out[..., 5, :] + keypoints_out[..., 6, :]) * 0.5
        # 7: Spine (Center between Pelvis and Thorax)
        h36m_kps[..., 7, :] = (h36m_kps[..., 0, :] + h36m_kps[..., 8, :]) * 0.5
        # 9: Neck (Proxy using the nose)
        h36m_kps[..., 9, :] = keypoints_out[..., 0, :]
        # 10: Head (Center between ears)
        h36m_kps[..., 10, :] = (keypoints_out[..., 1, :] + keypoints_out[..., 2, :]) * 0.5
        # 11: Left Shoulder
        h36m_kps[..., 11, :] = keypoints_out[..., 5, :]
        # 12: Left Elbow
        h36m_kps[..., 12, :] = keypoints_out[..., 7, :]
        # 13: Left Wrist
        h36m_kps[..., 13, :] = keypoints_out[..., 9, :]
        # 14: Right Shoulder
        h36m_kps[..., 14, :] = keypoints_out[..., 6, :]
        # 15: Right Elbow
        h36m_kps[..., 15, :] = keypoints_out[..., 8, :]
        # 16: Right Wrist
        h36m_kps[..., 16, :] = keypoints_out[..., 10, :]

        return h36m_kps

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
            
            if self.notation == 'H36M':
                keypoints_out = self.turn_into_h36m(keypoints_out)

        return keypoints_out  # (B,T,J,3)