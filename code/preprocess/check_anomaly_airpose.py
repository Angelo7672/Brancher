import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from torchinfo import summary
from datetime import datetime
import time
import argparse
import math

def audit_dataset(dataloader, root_idx=12):
    print(f"--- Inizio Audit Dataset (Root Index: {root_idx}) ---")
    
    global_max_abs = -float('inf')
    global_min_abs = float('inf')
    global_max_rel = -float('inf')
    
    # Per calcolare la lunghezza media degli "arti" (es. distanza tra giunti consecutivi)
    avg_dist_to_root = 0
    count = 0

    for i, batch in enumerate(dataloader):
        # Assumiamo batch["joints_3d"] shape (B, T, J, 3)
        gt = batch["joints_3d"]
        
        # 1. Analisi Valori Assoluti
        batch_max = gt.max().item()
        batch_min = gt.min().item()
        global_max_abs = max(global_max_abs, batch_max)
        global_min_abs = min(global_min_abs, batch_min)

        # 2. Analisi Valori Relativi
        root = gt[:, :, [root_idx], :]
        gt_rel = gt - root
        batch_max_rel = gt_rel.abs().max().item()
        global_max_rel = max(global_max_rel, batch_max_rel)

        # 3. Controllo "Giunti a Zero" (Spesso causa di outlier enormi)
        zeros = (gt == 0).all(dim=-1).sum().item()
        
        if i == 0:
            # Check specifico sul primo frame per capire le distanze
            dist_matrix = torch.norm(gt_rel[0, 0, :, :], dim=-1)
            avg_dist_to_root = dist_matrix.mean().item()

        if batch_max_rel > 100: # Se una persona è più grande di 100 metri
            print(f"![!] Batch {i}: Rilevata anomalia! Max Relativo = {batch_max_rel:.2f}")
            print(f"    Valore Root (X,Y,Z): {root[0,0,0,:].tolist()}")
            
    print("\n" + "="*40)
    print("RISULTATI ANALISI:")
    print(f"Range Assoluto:  [{global_min_abs:.2f}, {global_max_abs:.2f}]")
    print(f"Max Relativo:    {global_max_rel:.2f} (Distanza massima giunto-root)")
    print(f"Distanza Media:  {avg_dist_to_root:.4f} (Dovrebbe essere < 1.0 per metri)")
    
    # Interpretazione automatica
    if global_max_rel > 1000:
        print("\nDIAGNOSI: Probabile errore di unità di misura.")
        print("-> Se i valori sono ~3000, potrebbero essere MILLIMETRI o PIXEL.")
        print("-> Se la root è quasi 0, la sottrazione non sta spostando il centro.")
    elif global_max_rel < 0.001:
        print("\nDIAGNOSI: I dati sembrano piatti o compressi.")
    
    if (gt[:, :, root_idx, :] == 0).all():
        print("\nATTENZIONE: Il giunto scelto come ROOT è composto solo da ZERI.")
        print("La sottrazione non ha effetto.")
    print("="*40)


def forensic_analysis(dataloader, root_idx=12, threshold=3.0, uav=True):
    """
    Analisi granulare delle anomalie.
    uav=True: ignora i giunti 0-4 (faccia) nell'individuazione degli outlier.
    """
    label = "BODY ONLY" if uav else "FULL SKELETON"
    print(f"\n--- ANALISI FORENSE DETTAGLIATA ({label}, Soglia: {threshold}m) ---")
    
    found_any = False
    
    for b_idx, batch in enumerate(dataloader):
        gt_3d = batch["joints_3d"]  # (B, T, J, 3)
        
        # Calcoliamo la root-relative
        root = gt_3d[:, :, [root_idx], :]
        gt_rel = gt_3d - root
        
        # Distanza euclidea di ogni giunto dalla sua root
        dist_rel = torch.norm(gt_rel, dim=-1) # (B, T, J)
        
        # Se uav=True, mettiamo a zero le distanze dei giunti 0-4 per ignorarli
        if uav:
            # Creiamo un clone per non modificare i dati originali
            dist_to_check = dist_rel.clone()
            dist_to_check[:, :, 0:5] = 0.0 
        else:
            dist_to_check = dist_rel

        # Cerchiamo gli indici che superano la soglia
        anomalies = (dist_to_check > threshold).nonzero(as_tuple=False)
        
        if anomalies.size(0) > 0:
            found_any = True
            n_print = min(anomalies.size(0), 5)
            
            print(f"\n[!] Trovate {anomalies.size(0)} anomalie nel Batch {b_idx}")
            
            for i in range(n_print):
                b, t, j = anomalies[i].tolist()
                dist = dist_rel[b, t, j].item()
                
                coord_joint = gt_3d[b, t, j].tolist()
                coord_root = gt_3d[b, t, root_idx].tolist()
                
                print(f"    -> Giunto {j} | Distanza: {dist:.2f}m")
                print(f"       Root ({root_idx}): [{coord_root[0]:.2f}, {coord_root[1]:.2f}, {coord_root[2]:.2f}]")
                print(f"       Joint ({j}):  [{coord_joint[0]:.2f}, {coord_joint[1]:.2f}, {coord_joint[2]:.2f}]")
            
            if anomalies.size(0) > 5:
                print(f"    ... e altre {anomalies.size(0) - 5} anomalie in questo batch.")

    if not found_any:
        print(f"Nessun giunto{' (corpo)' if uav else ''} eccede la soglia di {threshold} metri.")
    print("\n" + "="*40 + "\n")


class AirPosePreprocessed(Dataset):
    def __init__(self, data_dir, frame_size):
        self.data_dir = data_dir
        self.width = float(frame_size[1])
        self.height = float(frame_size[0])
        self.samples = sorted(glob(os.path.join(data_dir, "block_*")))
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        pose2d_path = os.path.join(sample_dir, "keypoints2d.json")
        pose3d_path = os.path.join(sample_dir, "keypoints3d.json")

        # load 2D poses
        with open(pose2d_path, 'r') as f2:
            j2d = torch.tensor(json.load(f2), dtype=torch.float32)

        # load poses
        with open(pose3d_path, 'r') as f3:
            j3d = torch.tensor(json.load(f3), dtype=torch.float32)

        # ---------------------------------------------------------
        # 2D NORMALIZATION: From Pixel to [-1, 1]
        # ---------------------------------------------------------
        j2d[..., 0] = (j2d[..., 0] / self.width - 0.5) * 2
        j2d[..., 1] = (j2d[..., 1] / self.height - 0.5) * 2

        return {
            "joints_2d": j2d,     # (T, J, 3)
            "joints_3d": j3d     # (T, J, 3)
        }


def create_dataloaders(dataset, data_dir, batch_size, frame_size, num_workers, val_ratio, seed):
    if dataset == "sports":
        print("Using SportsPose dataset")
        full_ds = SportsPosePreprocessed(data_dir, frame_size)
    elif dataset == "air":
        print("Using AirPose dataset")
        full_ds = AirPosePreprocessed(data_dir, frame_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    total_len = len(full_ds)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    # Riproducibility
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset total: {total_len}")
    print(f"Train: {train_len} | Val: {val_len}")
    return train_loader, val_loader


train_loader, val_loader = create_dataloaders(
        'air',
        "/Experiments/angeloa00/AirPose_COCO/train",
        batch_size=32,
        frame_size=(352, 640),
        num_workers=4,
        val_ratio=0.2,
        seed=42
    )

# Esempio di utilizzo:
audit_dataset(train_loader, root_idx=12)
audit_dataset(val_loader, root_idx=12)

forensic_analysis(train_loader)
forensic_analysis(val_loader)