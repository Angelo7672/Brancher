import os
import torch
import torch.nn as nn
import argparse
import math
import json
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pickle

from model import create_model
from config import Config
from loss import loss_mpjpe, loss_p_mpjpe
from learning import load_pretrained_weights, AverageMeter

class AthletePosePreprocessed(Dataset):
    def __init__(self, data_dir):
        self.samples = sorted(glob(os.path.join(data_dir, "*.pkl")))
        
        if len(self.samples) == 0:
            raise ValueError(f"No .pkl files found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)
        
        j2d = torch.tensor(data['data_input'], dtype=torch.float32)
        j3d = torch.tensor(data['data_label'], dtype=torch.float32)
        
        # Ritorna sempre un dizionario con le chiavi standard
        sample = {
            "joints_2d": j2d,
            "joints_3d": j3d,
        }

        return sample

def create_test_dataloader(data_dir, batch_size, num_workers):
    print(f"Loading AthletePose Test Set from {data_dir}")
    test_ds = AthletePosePreprocessed(data_dir)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader

def test(model, loader, cfg):
    device = cfg.device
    model.eval()

    mpjpe_meter = AverageMeter()
    p_mpjpe_meter = AverageMeter()

    print(f"Starting Inference on {len(loader)} batches...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            gt = batch["joints_3d"].to(device)    # (B, T, J, 3)
            video = batch["joints_2d"].to(device) # (B, T, J, 3)

            # --- Forward ---
            pred = model(video)

            # --- Root-relative handling ---
            # Sottraiamo il root (indice 0) per calcolare l'errore relativo
            root_pred = pred[:, :, [0], :]
            root_gt   = gt[:, :, [0], :]

            pred_rel = (pred - root_pred)
            gt_rel = (gt - root_gt)

            # --- Calcolo Metriche ---
            # Nota: moltiplichiamo per 1000 se i dati sono in metri e vogliamo i mm
            err_mpjpe = loss_mpjpe(pred_rel, gt_rel)
            err_p_mpjpe = loss_p_mpjpe(pred_rel, gt_rel)

            mpjpe_meter.update(err_mpjpe.item(), gt.size(0))
            p_mpjpe_meter.update(err_p_mpjpe.item(), gt.size(0))

            if (i + 1) % 10 == 0:
                print(f" Batch {i+1}/{len(loader)} | MPJPE: {mpjpe_meter.avg*1000:.2f}mm")

    return {
        "mpjpe_mm": mpjpe_meter.avg * 1000,
        "p_mpjpe_mm": p_mpjpe_meter.avg * 1000
    }

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    checkpoint_path = "experiments/motionagformer/motionagformer-s-h36m.pth.tr"
    test_dir = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/test/frame_81/test"

    # 1. Caricamento Modello
    model = create_model(cfg.frame_size_sport)
    print(f"Loading checkpoint from: {checkpoint_path}")
  
    model = load_pretrained_weights(model, checkpoint_path, device)
    
    model.to(device)

    # 2. Setup Data
    test_loader = create_test_dataloader(
        test_dir,
        cfg.batch_size, 
        cfg.num_workers
    )

    # 3. Esecuzione Test
    results = test(model, test_loader, cfg)

    # 4. Logging Risultati
    output_str = (
        f"\n{'='*30}\n"
        f"TEST RESULTS\n"
        f"{'='*30}\n"
        f"Checkpoint: {os.path.basename(checkpoint_path)}\n"
        f"Total Samples: {len(test_loader.dataset)}\n"
        f"MPJPE:   {results['mpjpe_mm']:.4f} mm\n"
        f"P-MPJPE: {results['p_mpjpe_mm']:.4f} mm\n"
        f"{'='*30}\n"
    )
    
    print(output_str)

    # Salva i risultati in un file nella stessa cartella del checkpoint
    #res_path = os.path.join(os.path.dirname(checkpoint_path), "test_results.txt")
    #with open(res_path, "w") as f:
    #    f.write(output_str)
    #print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()