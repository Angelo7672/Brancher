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

            ys_hm = (idxs // Wh).float()
            xs_hm = (idxs % Wh).float()
            # TRASFORMAZIONE IN COORDINATE PIXEL GLOBALI
            # Spieghiamo il calcolo:
            # (xs_hm / Wh) -> posizione relativa nel crop (0-1)
            # * crop_w     -> posizione in pixel dentro il crop
            # + x1         -> posizione in pixel nel frame intero
            xs_global = x1 + (xs_hm * (crop_w / Wh))
            ys_global = y1 + (ys_hm * (crop_h / Hh))

            confs = confs.cpu().numpy()
            xs_g = xs_global.cpu().numpy()
            ys_g = ys_global.cpu().numpy()

            keypoints2d_list[t] = [[float(xs_g[j]), float(ys_g[j]), float(confs[j])] for j in range(J)]
    
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


def turn_into_h36m(keypoints_list):
    """
    Mapping da COCO (17 joints) a H36M (17 joints) lavorando su LISTE.
    keypoints_list: list di lunghezza T, ogni elemento è list di 17 joints [x, y, conf]
    """
    h36m_list = []
    
    for kp in keypoints_list:
        # Se il frame è vuoto (tutti zeri), inizializziamo h36m a zero
        if not kp or len(kp) == 0:
            h36m_list.append([[0.0, 0.0, 0.0] for _ in range(17)])
            continue
            
        # Creiamo un contenitore per i 17 nuovi joints
        new_kp = [[0.0, 0.0, 0.0] for _ in range(17)]
        
        # Helper per convertire in liste/tuple per facilità di accesso
        def get_j(idx): return kp[idx]

        # 0: Pelvis (Media tra le anche 11 e 12)
        l_hip, r_hip = get_j(11), get_j(12)
        new_kp[0] = [(l_hip[0]+r_hip[0])*0.5, (l_hip[1]+r_hip[1])*0.5, (l_hip[2]+r_hip[2])*0.5]
        
        # 1: Right Hip, 2: Right Knee, 3: Right Ankle
        new_kp[1], new_kp[2], new_kp[3] = get_j(12), get_j(14), get_j(16)
        
        # 4: Left Hip, 5: Left Knee, 6: Left Ankle
        new_kp[4], new_kp[5], new_kp[6] = get_j(11), get_j(13), get_j(15)
        
        # 8: Thorax (Media tra le spalle 5 e 6)
        l_sho, r_sho = get_j(5), get_j(6)
        new_kp[8] = [(l_sho[0]+r_sho[0])*0.5, (l_sho[1]+r_sho[1])*0.5, (l_sho[2]+r_sho[2])*0.5]
        
        # 7: Spine (Media tra Pelvis e Thorax)
        pelvis, thorax = new_kp[0], new_kp[8]
        new_kp[7] = [(pelvis[0]+thorax[0])*0.5, (pelvis[1]+thorax[1])*0.5, (pelvis[2]+thorax[2])*0.5]
        
        # 9: Neck (Proxy usando il naso 0)
        new_kp[9] = get_j(0)
        
        # 10: Head (Media tra le orecchie 1 e 2)
        l_ear, r_ear = get_j(1), get_j(2)
        new_kp[10] = [(l_ear[0]+r_ear[0])*0.5, (l_ear[1]+r_ear[1])*0.5, (l_ear[2]+r_ear[2])*0.5]
        
        # 11: Left Shoulder, 12: Left Elbow, 13: Left Wrist
        new_kp[11], new_kp[12], new_kp[13] = get_j(5), get_j(7), get_j(9)
        
        # 14: Right Shoulder, 15: Right Elbow, 16: Right Wrist
        new_kp[14], new_kp[15], new_kp[16] = get_j(6), get_j(8), get_j(10)
        
        h36m_list.append(new_kp)
        
    return h36m_list


# ---------------------------
# Main pipeline: iterate v2
# ---------------------------
def process_all(v2_dir, v3_dir, device='cuda', start_index=716):
    os.makedirs(v3_dir, exist_ok=True)
    extractor = KeypointExtractorYOLO(device=device)

    root_path = Path(v2_dir)
    all_video_files = sorted(list(root_path.glob("*.mp4")))
    video_files = all_video_files[start_index:]
    
    print(f"Trovati {len(video_files)} file .mp4 in {v2_dir}")
    print(f"Ripartenza dall'indice {start_index}. Video da elaborare: {len(video_files)}")

    for i, v_path in enumerate(video_files):
        if not os.path.isfile(v_path):
            print(f"[SKIP] video missing: {v_path}")
            continue
        v_path = str(v_path)
        folder_v3 = os.path.join(v3_dir, f"video_{i:05d}")
        os.makedirs(folder_v3, exist_ok=True)

        print(f"\nProcessing {v_path} ...")

        # read video frames (THWC uint8)
        video_frames, _, _ = read_video(v_path, output_format="THWC", pts_unit="sec")
        T = video_frames.shape[0]
        H_full = int(video_frames.shape[1])
        W_full = int(video_frames.shape[2])

        print(f"  Frames: {T}, Resolution: {H_full}x{W_full}")

        # detect all persons in the full video at once
        boxes_list = extractor.detect_persons(video_frames)

        # extract heatmaps & keypoints in one call
        heatmaps_full_list, keypoints_all, bbox_list = extractor.extract_from_boxes(video_frames, boxes_list)
        keypoints_all = turn_into_h36m(keypoints_all)

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
    V2_DIR = "/mnt/proj2/eu-25-52/ar17/DroneDataset/video_and_flipped"
    V3_DIR = "/mnt/proj2/eu-25-52/ar17/DroneDataset/train"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    process_all(V2_DIR, V3_DIR, device=DEVICE)

    
print("All videos processed.")