import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_video

from ultralytics import YOLO
from mmpose.apis import init_model
from config import Config

# ---------------------------
# 2D Pose Extractor (MMPose wrapper)
# ---------------------------
class PE(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=True):
        super().__init__()
        self.model = init_model(config_file, checkpoint_file, device=device)
        if freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        # x: [N, 3, H_model, W_model]
        feats = self.backbone(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        x = self.deconv_layer(feats)
        x = self.final_layer(x)
        return x  # [N, J, H_out, W_out]


# ---------------------------
# KeypointExtractor con YOLO + PE
# ---------------------------
class KeypointExtractorYOLO:
    def __init__(self, device='cuda'):
        cfg = Config()
        self.device = torch.device(device)
        # PE
        self.model_h, self.model_w = cfg.pe_size
        self.pe = PE(cfg.config_file, cfg.checkpoint_file, device=self.device).to(self.device)
        self.pe.eval()

        # YOLO
        self.detector = YOLO(cfg.yolo_version).to(self.device)
        self.detector.model.eval()

        # normalizer for MMPose inputs
        self.normalizer = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def detect_persons(self, frames_batch):
        """
        frames_batch: torch Tensor [N, H, W, 3] uint8 (0-255)
        returns: list length N, each element is tensor (K,4) xyxy of detected person(s) (may be empty tensor)
        NOTE: the ultralytics API accepts BCHW float in [0,1], but also supports tensors.
        We'll permute and normalize inside the call.
        """
        if len(frames_batch) == 0:
            return []

        batch = frames_batch.permute(0,3,1,2).float().to(self.device) / 255.0  # [N,3,H,W]

        with torch.no_grad():
            results = self.detector(batch, stream=False, verbose=False)

        all_boxes = []
        empty = torch.zeros((0,4), device=self.device)
        for res in results:
            if len(res.boxes) == 0:
                all_boxes.append(empty)
                continue
            # filter persons (class 0)
            mask = res.boxes.cls == 0
            if not mask.any():
                all_boxes.append(empty)
                continue
            boxes = res.boxes.xyxy[mask]  # (k,4) on device
            confs = res.boxes.conf[mask]
            # take single most confident box (single-pose assumption)
            _, idx = torch.topk(confs, k=1)
            all_boxes.append(boxes[idx])
        return all_boxes  # list[N] each (1,4) or empty tensor

    
    def extract_from_boxes(self, frames_batch, all_boxes):
        """
        frames_batch: torch Tensor [N, H, W, 3] uint8
        all_boxes: list length N of tensors (1,4) or empty

        Returns:
            heatmaps_crop_list: list length N of tensors [J, Hh, Wh] (float16, CPU)
            keypoints2d_list: list length N, each J×3 (global coords)
            bbox_list: list length N of [x1, y1, x2, y2] or [0,0,0,0] if missing
        """
        N = len(frames_batch)
        H_full = frames_batch.shape[1]
        W_full = frames_batch.shape[2]

        crops = []
        metas = []     # (frame_index, x1, y1, crop_h, crop_w)
        bbox_list = []

        # ==========================
        # 1) EXTRACT AND RESIZE CROPS
        # ==========================
        for t in range(N):
            boxes = all_boxes[t]

            if boxes.numel() == 0:
                bbox_list.append([0,0,0,0])
                continue

            # use first detection
            box = boxes[0].to(torch.int64)
            x1, y1, x2, y2 = map(int, box.tolist())

            # clamp to image
            x1 = max(0, min(x1, W_full - 1))
            x2 = max(0, min(x2, W_full))
            y1 = max(0, min(y1, H_full - 1))
            y2 = max(0, min(y2, H_full))

            if x2 <= x1 or y2 <= y1:
                bbox_list.append([0,0,0,0])
                continue

            bbox_list.append([x1, y1, x2, y2])

            crop = frames_batch[t, y1:y2, x1:x2]  # uint8

            crop_h, crop_w = crop.shape[:2]
            if crop_h == 0 or crop_w == 0:
                bbox_list[-1] = [0,0,0,0]
                continue

            # prepare for model
            crop_t = crop.permute(2,0,1).float() / 255.0
            resized = F.interpolate(
                crop_t.unsqueeze(0),
                size=(self.model_h, self.model_w),
                mode='bilinear',
                align_corners=False
            )[0]

            # normalize input
            resized = self.normalizer(resized)

            crops.append(resized)
            metas.append((t, x1, y1, crop_h, crop_w))

        # If no valid crops
        if len(crops) == 0:
            heatmaps_crop_list = [torch.zeros((self.num_joints, self.model_h, self.model_w),
                                              dtype=torch.float16)] * N
            keypoints2d_list = [[[0.0,0.0,0.0] for _ in range(self.num_joints)] for _ in range(N)]
            return heatmaps_crop_list, keypoints2d_list, bbox_list

        # ==========================
        # 2) RUN KEYPOINT EXTRACTOR (PE) ON THE CROPS
        # ==========================
        crop_batch = torch.stack(crops).to(self.device)  # [M,3,Hm,Wm]

        with torch.no_grad():
            heatmaps_batch = self.pe(crop_batch)  # [M,J,Hh,Wh]

        M, J, Hh, Wh = heatmaps_batch.shape
        hmap_norm = torch.clamp(heatmaps_batch, min=0.0, max=1.0) # Forzo il max a 1.0

        # ==========================
        # 3) PREPARE OUTPUT LISTS
        # ==========================
        heatmaps_crop_list = [None] * N
        keypoints2d_list = [[] for _ in range(N)]

        # ==========================
        # 4) PER-CROP PROCESSING
        # ==========================
        for i in range(M):
            t, x1, y1, crop_h, crop_w = metas[i]
            hm_local = hmap_norm[i]  # [J,Hh,Wh]

            # === SAVE **CROP HEATMAP**, float16 CPU ===
            heatmaps_crop_list[t] = hm_local.detach().cpu().half()

            # === COMPUTE KEYPOINTS (in LOCAL HEATMAP RESOLUTION) ===
            hm_flat = hm_local.view(J, -1)
            confs, idxs = torch.max(hm_flat, dim=1)

            ys = (idxs // Wh).cpu().numpy()
            xs = (idxs % Wh).cpu().numpy()
            confs = confs.cpu().numpy()

            # upscale local heatmap coords → crop coords
            scale_x = crop_w / float(Wh)
            scale_y = crop_h / float(Hh)

            kps = []
            for j in range(J):
                x_local = xs[j]
                y_local = ys[j]

                # map to global coords
                x_global = float(x_local * scale_x + x1)
                y_global = float(y_local * scale_y + y1)

                kps.append([x_global, y_global, float(confs[j])])

            keypoints2d_list[t] = kps

        # ==========================
        # 5) FILL MISSING FRAMES
        # ==========================
        # === Fill missing frames using previous keypoints ===
        prev_kp = None

        for t in range(N):
            if heatmaps_crop_list[t] is None:
                # heatmap zero
                heatmaps_crop_list[t] = torch.zeros((J, Hh, Wh), dtype=torch.float16)

                # keypoints: copy last valid OR zeros if none
                if prev_kp is not None:
                    keypoints2d_list[t] = [[x, y, 0.0] for (x, y, _) in prev_kp]
                else:
                    keypoints2d_list[t] = [[0.0, 0.0, 0.0] for _ in range(J)]
            else:
                # update last known valid keypoints
                prev_kp = keypoints2d_list[t]
        
        return heatmaps_crop_list, keypoints2d_list, bbox_list


# ---------------------------
# Main pipeline: iterate v2
# ---------------------------
def process_all(v2_dir, v3_dir, device='cuda'):
    os.makedirs(v3_dir, exist_ok=True)
    extractor = KeypointExtractorYOLO(device=device)

    folders = sorted([d for d in os.listdir(v2_dir) if os.path.isdir(os.path.join(v2_dir, d))])
    print(f"Found {len(folders)} folders in {v2_dir}")

    for folder in folders:
        folder_v2 = os.path.join(v2_dir, folder)
        folder_v3 = os.path.join(v3_dir, folder)
        os.makedirs(folder_v3, exist_ok=True)

        video_path = os.path.join(folder_v2, "video.mp4")

        if not os.path.isfile(video_path):
            print(f"[SKIP] video missing: {video_path}")
            continue

        print(f"\nProcessing {folder} ...")

        # read video frames (THWC uint8)
        video_frames, _, _ = read_video(video_path, output_format="THWC", pts_unit="sec")
        T = video_frames.shape[0]
        H_full = int(video_frames.shape[1])
        W_full = int(video_frames.shape[2])

        print(f"  Frames: {T}, Resolution: {H_full}x{W_full}")

        # detect all persons in the full video at once
        boxes_list = extractor.detect_persons(video_frames)

        # extract heatmaps & keypoints in one call
        heatmaps_full_list, keypoints_all, bbox_list = extractor.extract_from_boxes(video_frames, boxes_list)

        # stack heatmaps -> [T, J, H, W] (float16 CPU)
        heatmaps_stack = torch.stack(heatmaps_full_list, dim=0)

        # save heatmaps
        heatmap_path = os.path.join(folder_v3, "heatmaps.pt")
        torch.save(heatmaps_stack, heatmap_path)
        print(f"  Saved heatmaps: {heatmap_path} (shape {tuple(heatmaps_stack.shape)}, dtype={heatmaps_stack.dtype})")

        # save keypoints2d.json
        kp_path = os.path.join(folder_v3, "keypoints2d.json")
        with open(kp_path, "w") as f:
            json.dump(keypoints_all, f)
        print(f"  Saved keypoints2d json: {kp_path}")

        # save bboxes.json
        bbox_path = os.path.join(folder_v3, "bboxes.json")
        with open(bbox_path, "w") as f:
            json.dump(bbox_list, f)
        print(f"  Saved bboxes json: {bbox_path}")

    print("\nAll done.")


# ---------------------------
# run as script
# ---------------------------
if __name__ == "__main__":
    V2_DIR = "/Experiments/angeloa00/AirPose/test"
    V3_DIR = "/Experiments/angeloa00/AirPose_COCO/test"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    process_all(V2_DIR, V3_DIR, device=DEVICE)

    
print("All videos processed.")