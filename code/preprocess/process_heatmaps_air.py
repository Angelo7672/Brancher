#!/usr/bin/env python3
"""
prepare_dataset_yolo.py

Processa tutti i video in v2_dir:
 - rileva persona con YOLO (batch/chunk per limitare VRAM)
 - esegue MMPose (PE) sulle crop
 - rimappa heatmap alla risoluzione frame originale
 - salva heatmaps.pt (torch.float16) e keypoints2d.json
 - copia pose3d.json nella cartella corrispondente di v3_dir
"""

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
        # detector.model may already be on device but ensure
        try:
            self.detector.model.to(self.device)
        except Exception:
            pass
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
            heatmaps_full_list: list length N of tensors [J, H, W] (on CPU, float16)
            keypoints2d_list: list length N of list of J (x,y,conf) tuples or empty list
        """
        N = len(frames_batch)
        H_full = frames_batch.shape[1]
        W_full = frames_batch.shape[2]

        crops = []
        metas = []  # list of (frame_index, x1, y1, crop_h, crop_w)

        # collect crops
        for t in range(N):
            boxes = all_boxes[t]
            if boxes.numel() == 0:
                continue
            # use first box only
            box = boxes[0].to(torch.int64)
            x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
            # clamp to frame
            x1 = max(0, min(x1, W_full-1))
            x2 = max(0, min(x2, W_full))
            y1 = max(0, min(y1, H_full-1))
            y2 = max(0, min(y2, H_full))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frames_batch[t, y1:y2, x1:x2]  # uint8
            if crop.numel() == 0:
                continue
            crop_h, crop_w = crop.shape[:2]
            crop_t = crop.permute(2,0,1).float() / 255.0  # [3, crop_h, crop_w]
            # resize to model size
            resized = F.interpolate(crop_t.unsqueeze(0), size=(self.model_h, self.model_w),
                                    mode='bilinear', align_corners=False)[0]
            # normalize
            resized = self.normalizer(resized)
            crops.append(resized)
            metas.append((t, x1, y1, crop_h, crop_w))

        if len(crops) == 0:
            # return lists of None/empty for all frames
            heatmaps_full_list = [None] * N
            keypoints2d_list = [[] for _ in range(N)]
            return heatmaps_full_list, keypoints2d_list

        crop_batch = torch.stack(crops).to(self.device)  # [M, 3, Hm, Wm]

        with torch.no_grad():
            heatmaps_batch = self.pe(crop_batch)  # [M, J, Hh, Wh]

        M, J, Hh, Wh = heatmaps_batch.shape

        # prepare outputs
        heatmaps_full_list = [None] * N
        keypoints2d_list = [[] for _ in range(N)]

        # for each produced heatmap, remap back to full frame and extract keypoints
        for i in range(M):
            t, x1, y1, crop_h, crop_w = metas[i]
            hm_local = heatmaps_batch[i]  # [J, Hh, Wh]

            # resize hm_local to crop size
            hm_resized = F.interpolate(hm_local.unsqueeze(0), size=(crop_h, crop_w),
                                       mode='bilinear', align_corners=False)[0]  # [J, crop_h, crop_w]

            # full canvas
            hm_full = torch.zeros((J, H_full, W_full), dtype=hm_resized.dtype, device=hm_resized.device)
            # place in position
            y2 = y1 + crop_h
            x2 = x1 + crop_w
            hm_full[:, y1:y2, x1:x2] = hm_resized

            # move to CPU half precision for saving to limit memory
            hm_full_cpu = hm_full.detach().cpu().half()  # float16 on CPU
            heatmaps_full_list[t] = hm_full_cpu

            # extract keypoints (per-joint max)
            # flatten per joint
            hm_flat = hm_local.view(J, -1)  # on device
            confs, idxs = torch.max(hm_flat, dim=1)  # [J]
            # compute x,y in heatmap local resolution Hh x Wh
            ys = (idxs // Wh).cpu().numpy()
            xs = (idxs % Wh).cpu().numpy()
            confs = confs.cpu().numpy()

            # scale from heatmap local (Hh,Wh) -> original crop -> global coords
            # scale factors
            scale_x = float(crop_w) / float(Wh)
            scale_y = float(crop_h) / float(Hh)
            keypoints = []
            for j in range(J):
                x_local = xs[j]
                y_local = ys[j]
                x_global = float(x_local * scale_x + x1)
                y_global = float(y_local * scale_y + y1)
                conf = float(confs[j])
                keypoints.append([x_global, y_global, conf])
            keypoints2d_list[t] = keypoints

        # fill missing frames with zeros / empty
        for t in range(N):
            if heatmaps_full_list[t] is None:
                # no detection for this frame -> zero heatmap
                heatmaps_full_list[t] = torch.zeros((J, H_full, W_full), dtype=torch.float16)
                keypoints2d_list[t] = [[0.0, 0.0, 0.0] for _ in range(J)]

        return heatmaps_full_list, keypoints2d_list


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
        heatmaps_full_list, keypoints_all = extractor.extract_from_boxes(video_frames, boxes_list)

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

    print("\nAll done.")


# ---------------------------
# run as script
# ---------------------------
if __name__ == "__main__":
    # CONFIG: cambia questi path se serve
    V2_DIR = "/Experiments/angeloa00/AirPose/train"
    V3_DIR = "/Experiments/angeloa00/AirPose_COCO/train"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    process_all(V2_DIR, V3_DIR, device=DEVICE)

    
print("All videos processed.")