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
import numpy as np

from model import create_model
from learning import load_pretrained_weights
from config import Config


# ------------------------
# Loss functions
# ------------------------
def loss_mpjpe(predicted, target, exclude=[0, 1, 2, 3, 4]):
    """
    MPJPE calcolato escludendo alcuni keypoints (default: COCO nose, eyes, ears).
    predicted/target: (B, T, J, 3) oppure (N, J, 3)
    """
    assert predicted.shape == target.shape

    J = predicted.shape[-2]  # number of joints
    device = predicted.device

    # mask: 1 = included, 0 = excluded
    mask = torch.ones(J, dtype=torch.bool, device=device)
    mask[exclude] = False

    # Applica la mask sui joint
    pred_sel = predicted[..., mask, :]
    targ_sel = target[..., mask, :]

    # MPJPE standard su joints filtrati
    return torch.mean(torch.norm(pred_sel - targ_sel, dim=-1))


# ------------------------
# Utility
# ------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
        #video_path = os.path.join(sample_dir, "video.mp4")
        pose2d_path = os.path.join(sample_dir, "pose2d.json")
        #heatmap_path = os.path.join(sample_dir, "heatmaps.pt")
        pose3d_path = os.path.join(sample_dir, "pose3d.json")

        # ✅ Legge il video con torchvision (frames in RGB)
        #frames, _, _ = read_video(video_path, pts_unit="sec")

        # load 2D poses
        with open(pose2d_path, 'r') as f2:
            data = json.load(f2)
            # data è direttamente una lista: (T, J, 3) dove T=frame, J=joint, 3=[x,y,conf]
            keypoints_2d = data["keypoints"]      # <-- FONDAMENTALE
            j2d = torch.tensor(keypoints_2d, dtype=torch.float32)

        # Load heatmaps (carica su CPU per evitare problemi CUDA in worker processes)
        #heatmaps = torch.load(heatmap_path, map_location='cpu')

        # load poses
        with open(pose3d_path, 'r') as f3:
            j3d = torch.tensor(json.load(f3), dtype=torch.float32)

        # ---------------------------------------------------------
        # 1. NORMALIZZAZIONE 2D: Da Pixel a [-1, 1]
        # ---------------------------------------------------------
        # Formula: (valore / dimensione - 0.5) * 2
        # Esempio: pixel 0 -> -1, pixel 1920 -> 1
        j2d[..., 0] = (j2d[..., 0] / self.width - 0.5) * 2
        j2d[..., 1] = (j2d[..., 1] / self.height - 0.5) * 2

        return {
            #"video": frames,      # (T, H, W, C)
            "joints_2d": j2d,     # (T, J, 2)
            #"heatmaps": heatmaps,   # (T, J, H, W)
            "joints_3d": j3d     # (T, J, 3)
        }

def create_dataloaders(data_dir, batch_size, frame_size, num_workers, val_ratio, seed):
    full_ds = AirPosePreprocessed(data_dir, frame_size)
    total_len = len(full_ds)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    # Split riproducibile
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

    print(f"Dataset totale: {total_len}")
    print(f"Train: {train_len} | Val: {val_len}")
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scaler, loss_meter, cfg: Config):
    device = cfg.device
    model.train()

    print("I'm trainining")
    batch_idx = 0

    # views present in the loader (derived from first batch)
    # but we'll fetch views per-batch to be robust

    for batch in loader:    
        #print(f" Batch {batch_idx+1}/{len(loader)}", end='\r')
        print(f"Batch {batch_idx+1}/{len(loader)}", end='\r', flush=True)
        batch_idx += 1

        # {
        #     "element": [
        #         {
        #             "video": tensor(T, H, W, C),
        #             "joints_2d": tensor(T, J, 2),
        #             "joints_3d": tensor(T, J, 3),
        #             "metadata": {...}
        #         }
        #     ]
        # }

        gt = batch["joints_3d"].to(device)
        #video = batch["video"].to(device)  # (B, T, H, W, C)
        video = batch["joints_2d"].to(device)  # (B, T, J, 3)
        #video = batch["heatmaps"].to(device)    # (B, T, J, H, W)

        # --- Azzera i gradienti ---
        optimizer.zero_grad()

        # --- Forward pass unico (usa AMP se abilitato) ---
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(video)  # (B, T, J, 3)


            # subito dopo pred = model(video)
            if not torch.isfinite(pred).all():
                print("NON FINITO: pred contiene NaN o Inf", torch.isnan(pred).sum().item(), torch.isinf(pred).sum().item())
            if not torch.isfinite(gt).all():
                print("NON FINITO: gt contiene NaN o Inf", torch.isnan(gt).sum().item(), torch.isinf(gt).sum().item())


            # --- Root-relative handling ---
            # pelvis = average of joints 11 and 12
            pelvis_gt = gt[:, :, [11,12], :].mean(dim=2, keepdim=True)
            pelvis_pred = pred[:, :, [11,12], :].mean(dim=2, keepdim=True)
            pred_rel = pred - pelvis_pred
            gt_rel = gt - pelvis_gt

            # --- Compute losses ---
            loss = loss_mpjpe(pred_rel, gt_rel)

        # --- Backward + step ---
        scaler.scale(loss).backward()


        # prima di scaler.step(optimizer)
        for n, p in model.named_parameters():
            if p.grad is None: continue
            if not torch.isfinite(p.grad).all():
                print("GRAD NaN/Inf in", n, "sum:", torch.isnan(p.grad).sum().item(), "inf:", torch.isinf(p.grad).sum().item())
                break

        
        scaler.step(optimizer)
        scaler.update()

        # --- Update dei meter ---
        loss_meter.update(loss.item(), gt.size(0))


def evaluate(model, loader, cfg: Config):
    device = cfg.device
    model.eval()

    print("I'm evaluating")
    batch_idx = 0

    mpjpe_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            #print(f" Batch {batch_idx+1}/{len(loader)}", end='\r')
            print(f"Batch {batch_idx+1}/{len(loader)}", end='\r', flush=True)
            batch_idx += 1

            gt = batch["joints_3d"].to(device)  # (B, T, J, 3)
            #video = batch["video"].to(device)   # (B, T, H, W, C)
            video = batch["joints_2d"].to(device)  # (B, T, J, 3)
            #video = batch["heatmaps"].to(device)    # (B, T, J, H, W)

            # --- Forward ---
            pred = model(video)  # (B, T, J, 3)
    
    
            # subito dopo pred = model(video)
            if not torch.isfinite(pred).all():
                print("NON FINITO: pred contiene NaN o Inf", torch.isnan(pred).sum().item(), torch.isinf(pred).sum().item())
            if not torch.isfinite(gt).all():
                print("NON FINITO: gt contiene NaN o Inf", torch.isnan(gt).sum().item(), torch.isinf(gt).sum().item())
    
    
    
            # --- Root-relative using pelvis ---
            pelvis_pred = pred[:, :, [11, 12], :].mean(dim=2, keepdim=True)
            pelvis_gt   = gt[:, :, [11, 12], :].mean(dim=2, keepdim=True)

            pred_rel = pred - pelvis_pred
            gt_rel = gt - pelvis_gt

            err1 = loss_mpjpe(pred_rel, gt_rel)

            mpjpe_meter.update(err1.item(), gt.size(0))

    results = {"mpjpe": mpjpe_meter.avg}
    return results


# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--finetune_from", type=str, default=None, help="Directory containing best_weight.pth to finetune from")
    
    args = parser.parse_args()

    if args.finetune_from is not None and args.resume:
        raise ValueError("Non puoi usare --finetune_from e --resume insieme")
    elif args.finetune_from is None and not args.resume:
        raise ValueError("Devi specificare almeno --finetune_from o --resume")

    cfg = Config()
    device = cfg.device

    print("Impostazione dataset e dataloader...")

    # Datasets
    train_loader, val_loader = create_dataloaders(
        cfg.datapath,
        batch_size=cfg.batch_size,
        frame_size=cfg.frame_size,
        num_workers=cfg.num_workers,
        val_ratio=0.2,
        seed=cfg.seed
    )

    # === Gestione directory exp ===
    if args.finetune_from:
        # folder passed by user → contains best_weight.pth
        ft_source_dir = args.finetune_from
        weight_path = os.path.join(ft_source_dir, "best_weight.pth")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"best_weight.pth non trovato in {ft_source_dir}")

        print(f"Carico pesi iniziali da: {weight_path}")

        checkpoint = torch.load(weight_path, map_location=device)
        start_epoch = 0
        best_val = float("inf")

    else args.resume and args.resume_path:
        exp_dir = os.path.dirname(args.resume_path)

        best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")
        log_path = os.path.join(exp_dir, "log.txt")


    # Create model
    model = create_model()
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # AMP scaler, Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Simple linear warmup.
    # Idea: partire con LR piccolo e aumentarlo gradualmente per i primi warmup_epochs.
    # Warmup + Cosine Decay scheduler
    def lr_lambda(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            # Warmup lineare: da 0 → 1
            return float(epoch) / float(max(1, cfg.warmup_epochs))
        else:
            # Cosine decay per il resto del training
            progress = (epoch - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
            return 0.5 * (1 + np.cos(np.pi * progress))  # da 1 → 0
    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) # scheduler che applica lr_lambda ad ogni step.

    # In pratica:
    #    AdamW aggiorna i pesi.
    #    AMP accelera il training con mixed precision.
    #    Warmup evita di “spaventare” la rete con un LR troppo alto all’inizio.


    start_epoch = 0
    best_val = float("inf")

    if args.finetune_from:
        model = load_pretrained_weights(model, checkpoint)

    elif args.resume and args.resume_path is not None and os.path.isfile(args.resume_path):
        print(f"Riprendo il training da {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=cfg.device)
        start_epoch = checkpoint["epoch"] + 1

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))

        best_val = checkpoint.get("val_mpjpe", float("inf"))
        print(f"Ripreso da epoch {start_epoch}, best val MPJPE = {best_val:.2f}")

    model = model.to(device)

    # === Setup logging ===
    log_path = os.path.join(exp_dir, "log.txt")

    with open(log_path, "a") as f:
        if args.finetune_from:
            f.write("\n==== Fine-tune start ====\n")
            f.write(f"Loaded pretrained from: {weight_path}\n")
        elif args.resume:
            f.write("\n==== Fine-tune resume ====\n")

    # Training
    total_epochs = cfg.epochs

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{total_epochs}")

        losses = AverageMeter()

        train_one_epoch(model, train_loader, optimizer, scaler, losses, cfg)

        va = evaluate(model, val_loader, cfg)
        scheduler.step()  # step del LR scheduler

        # === Salvataggio di sicurezza ===
        epoch_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch+1}_ft.pth")
        torch.save({
            'cfg': cfg.__dict__,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'val_mpjpe': best_val,
        }, epoch_ckpt_path)
        print(f"Checkpoint salvato: {epoch_ckpt_path}")

        # 🔹 Elimina il file di due epoche fa
        old_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch-1}_ft.pth")
        if os.path.exists(old_ckpt_path):
            os.remove(old_ckpt_path)
            print(f"Checkpoint vecchio rimosso: {old_ckpt_path}")

        epoch_time_hours = (time.time() - epoch_start) / 3600.0
        current_lr = optimizer.param_groups[0]['lr']

        log_line = (
            f"Epoch {epoch+1:02d}/{total_epochs} | "
            f"LR {current_lr:.9f} | "
            f"train loss {losses.avg * 1000:.4f} mm | "
            f"val mpjpe {va['mpjpe'] * 1000:.4f} mm | "
            f"time {epoch_time_hours * 60:.2f}min"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # save best
        best_ckpt_ft_path = os.path.join(exp_dir, "best_weight_ft.pth")
        if va['mpjpe'] < best_val:
            best_val = va['mpjpe']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'cfg': cfg.__dict__,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'val_mpjpe': best_val,
            }, best_ckpt_ft_path)
            print(f"Saved best checkpoint (val MPJPE={best_val:.2f})")

if __name__ == "__main__":
    main()