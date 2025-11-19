import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sportspose.dataset import SportsPoseDataset
import cv2


def resize_joints_2d(joints, orig_h, orig_w, target_size):
    """Ridimensiona le coordinate 2D in base alla nuova risoluzione."""
    target_h, target_w = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    joints_resized = joints.clone()
    joints_resized[..., 0] *= scale_x
    joints_resized[..., 1] *= scale_y
    return joints_resized


def save_video_mp4(video_tensor, path, fps=90):
    """
    Salva un video torch (T, H, W, C) in formato mp4 (H.264).
    """
    video_np = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
    T, H, W, C = video_np.shape

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    
    for t in range(T):
        frame = cv2.cvtColor(video_np[t], cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()


def process_view(elem, view, target_size):
    """Elabora una singola vista del sample."""
    # Video (T, H, W, C)
    v = elem["video"]["image"][view]
    v = torch.from_numpy(v).permute(0, 3, 1, 2)  # (T, C, H, W)
    v = F.interpolate(v, size=target_size, mode="bilinear", align_corners=False)
    v = v.permute(0, 2, 3, 1) / 255.0  # (T, H, W, C), normalizzato 0-1

    # Joints 2D
    j = elem["joints_2d"][view]
    j = torch.as_tensor(j, dtype=torch.float32)
    H, W = elem["video"]["image"][view].shape[1:3]
    j = resize_joints_2d(j, H, W, target_size)

    # Joints 3D (uguali per tutte le viste)
    joints_3d = torch.as_tensor(elem["joints_3d"]["data_points"], dtype=torch.float32)

    return v, j, joints_3d


def preprocess_dataset(
    dataset,
    output_dir,
    target_size,
    start_idx=0,
    end_idx=None,
):
    os.makedirs(output_dir, exist_ok=True)
    n = len(dataset) if end_idx is None else min(end_idx, len(dataset))

    for idx in tqdm(range(start_idx, n)):
        try:
            elem = dataset[idx]

            # Processa ogni view separatamente
            for view_name in elem["video"]["image"]:
                view_dir = os.path.join(output_dir, f"sample_{idx:05d}_view_{view_name}")
                video_path = os.path.join(view_dir, "video.mp4")
                pose2d_path = os.path.join(view_dir, "pose2d.json")
                pose3d_path = os.path.join(view_dir, "pose3d.json")

                # Resume automatico
                if os.path.exists(video_path) and os.path.exists(pose2d_path) and os.path.exists(pose3d_path):
                    continue

                os.makedirs(view_dir, exist_ok=True)

                video, joints_2d, joints_3d = process_view(elem, view_name, target_size)

                # Salva video
                save_video_mp4(video, video_path)

                # Salva pose in JSON
                json.dump(joints_2d.tolist(), open(pose2d_path, "w"))
                json.dump(joints_3d.tolist(), open(pose3d_path, "w"))

        except Exception as e:
            print(f"[!] Errore su elemento {idx}: {e}")
            continue


if __name__ == "__main__":
    train_ds = SportsPoseDataset(
        data_dir="/mnt/proj2/eu-25-52/ar17/SportsPose",
        sample_level="video",
        seq_size=243,
    )

    preprocess_dataset(
        train_ds,
        output_dir="/mnt/proj2/eu-25-52/ar17/SportsPose_reprocessed_v2",
        target_size=(640, 416),
        start_idx=0,
        end_idx=None,
    )
