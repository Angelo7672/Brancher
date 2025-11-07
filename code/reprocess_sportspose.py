import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sportspose.dataset import SportsPoseDataset


def resize_joints_2d(joints, orig_h, orig_w, target_size):
    """Ridimensiona le coordinate 2D in base alla nuova risoluzione."""
    target_h, target_w = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    joints_resized = joints.clone()
    joints_resized[..., 0] *= scale_x
    joints_resized[..., 1] *= scale_y
    return joints_resized


def process_view(elem, view, target_size):
    """Elabora una singola vista del sample."""
    # Video (T, H, W, C)
    v = elem["video"]["image"][view]
    v = torch.from_numpy(v).permute(0, 3, 1, 2)  # (T, C, H, W)
    v = F.interpolate(v, size=target_size, mode="bilinear", align_corners=False)
    v = v.permute(0, 2, 3, 1)  # (T, H, W, C)

    # Joints 2D
    j = elem["joints_2d"][view]
    j = torch.as_tensor(j, dtype=torch.float32)

    H, W = elem["video"]["image"][view].shape[1:3]
    j = resize_joints_2d(j, H, W, target_size)

    # Joints 3D (uguali per tutte le viste)
    joints_3d = torch.as_tensor(elem["joints_3d"]["data_points"], dtype=torch.float32)

    return {
        "element": [
            {
                "video": v,
                "joints_2d": j,
                "joints_3d": joints_3d,
            }
        ]
    }


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
                out_name = f"sample_{idx:05d}_view_{view_name}.pt"
                out_path = os.path.join(output_dir, out_name)

                # Se già esiste, salta (resume automatico)
                if os.path.exists(out_path):
                    continue

                processed = process_view(elem, view_name, target_size)
                torch.save(processed, out_path)

        except Exception as e:
            print(f"[!] Errore su elemento {idx}: {e}")
            continue


if __name__ == "__main__":

    train_ds = SportsPoseDataset(
        data_dir="/DATA/SportsPose",
        sample_level="video",
        seq_size=243,
    )

    preprocess_dataset(
        train_ds,
        output_dir="/Experiments/angeloa00/SportsPose",
        target_size=(640, 416),
        start_idx=0,   # puoi ripartire da qui
        end_idx=1,  # opzionale
    )
