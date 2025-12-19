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

from model import create_model
from config import Config
from loss import *
from learning import AverageMeter


# ------------------------
# Utility
# ------------------------
class SportsPosePreprocessed(Dataset):
    def __init__(self, data_dir, frame_size):
        self.data_dir = data_dir
        self.width = float(frame_size[1])
        self.height = float(frame_size[0])
        self.samples = sorted(glob(os.path.join(data_dir, "sample_*_view_*")))
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
            j2d = torch.tensor(json.load(f2), dtype=torch.float32)

        # Load heatmaps (carica su CPU per evitare problemi CUDA in worker processes)
        #heatmaps = torch.load(heatmap_path, map_location='cpu')

        # load poses
        with open(pose3d_path, 'r') as f3:
            j3d = torch.tensor(json.load(f3), dtype=torch.float32)

        # ---------------------------------------------------------
        # 1. NORMALIZZAZIONE 2D: Da Pixel a [-1, 1]
        # ---------------------------------------------------------
        j2d[..., 0] = (j2d[..., 0] / self.width - 0.5) * 2
        j2d[..., 1] = (j2d[..., 1] / self.height - 0.5) * 2

        return {
            #"video": frames,      # (T, H, W, C)
            "joints_2d": j2d,     # (T, J, 3)
            #"heatmaps": heatmaps,   # (T, J, H, W)
            "joints_3d": j3d     # (T, J, 3)
        }

def create_dataloaders(data_dir, batch_size, frame_size, num_workers, val_ratio, seed):
    full_ds = SportsPosePreprocessed(data_dir, frame_size)
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

def adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, target_lr):
    """Gestisce solo le prime N epoche di warmup lineare"""
    if epoch < warmup_epochs:
        lr = target_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return True # Warmup attivo
    return False # Warmup finito

def decay_lr_exponentially(lr, optimizer, lr_decay=0.99):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr

# ------------------------


def train_one_epoch(model, loader, optimizer, cfg: Config):
    device = cfg.device
    model.train()

    print("I'm trainining")
    batch_idx = 0

    mpjpe_meter = AverageMeter()
    bone_meter = AverageMeter()
    velocity_meter = AverageMeter()
    acceleration_meter = AverageMeter()

    for batch in loader:    
        print(f" Batch {batch_idx+1}/{len(loader)}", end='\r')
        batch_idx += 1

        # {
        #     "element": [
        #         {
        #             "video": tensor(T, H, W, C),
        #             "joints_2d": tensor(T, J, 3),
        #             "joints_3d": tensor(T, J, 3),
        #             "metadata": {...}
        #         }
        #     ]
        # }

        gt = batch["joints_3d"].to(device)
        #video = batch["video"].to(device)  # (B, T, H, W, C)
        video = batch["joints_2d"].to(device)  # (B, T, J, 3)
        #video = batch["heatmaps"].to(device)    # (B, T, J, H, W)

        optimizer.zero_grad()   # azzera i gradienti

        pred = model(video)  # (B, T, J, 3)
        
        # subito dopo pred = model(video)
        if not torch.isfinite(pred).all():
            print("NON FINITO: pred contiene NaN o Inf", torch.isnan(pred).sum().item(), torch.isinf(pred).sum().item())
        if not torch.isfinite(gt).all():
            print("NON FINITO: gt contiene NaN o Inf", torch.isnan(gt).sum().item(), torch.isinf(gt).sum().item())
        
        # --- Root-relative handling ---
        root_gt = gt[:, :, [12], :]
        gt_rel = (gt - root_gt) / cfg.scale_factor  # normalizza a [-1, 1] per Tanh

        # --- Compute losses ---
        mpjpe = loss_mpjpe(pred, gt_rel)
        bone = loss_limb_var(pred)
        velocity = velocity_smoothness_loss(pred)
        acceleration = acceleration_smoothness_loss(pred)
        loss = mpjpe #+ cfg.bone_loss_weight * bone + cfg.velocity_loss_weight * velocity + cfg.acceleration_loss_weight * acceleration
        mpjpe_meter.update(mpjpe.item(), gt.size(0))
        bone_meter.update(bone.item(), gt.size(0))
        velocity_meter.update(velocity.item(), gt.size(0))
        acceleration_meter.update(acceleration.item(), gt.size(0))

        # --- Backward + step ---
        loss.backward()


        # prima di scaler.step(optimizer)
        for n, p in model.named_parameters():
            if p.grad is None: continue
            if not torch.isfinite(p.grad).all():
                print("GRAD NaN/Inf in", n, "sum:", torch.isnan(p.grad).sum().item(), "inf:", torch.isinf(p.grad).sum().item())
                break

        
        optimizer.step()

    results = {"mpjpe": mpjpe_meter.avg, "bone": bone_meter.avg, "velocity": velocity_meter.avg, "acceleration": acceleration_meter.avg}
    
    return results


def evaluate(model, loader, cfg: Config):
    device = cfg.device
    model.eval()

    print("I'm evaluating")
    batch_idx = 0

    mpjpe_meter = AverageMeter()
    p_mpjpe_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            print(f" Batch {batch_idx+1}/{len(loader)}", end='\r')
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
    
    
    
            # --- Root-relative ---
            root_pred = pred[:, :, [12], :]
            root_gt   = gt[:, :, [12], :]

            pred_rel = (pred - root_pred) * cfg.scale_factor # riporta a metri
            gt_rel = gt - root_gt

            err1 = loss_mpjpe(pred_rel, gt_rel)
            err2 = loss_p_mpjpe(pred_rel, gt_rel)

            mpjpe_meter.update(err1.item(), gt.size(0))
            p_mpjpe_meter.update(err2.item(), gt.size(0))

    results = {"mpjpe": mpjpe_meter.avg, "p_mpjpe": p_mpjpe_meter.avg}
    return results


# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()

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

    # Create or get experiment directory
    if args.resume and args.resume_path:
        # Extract experiment directory from checkpoint path by removing the weight filename
        exp_dir = os.path.dirname(args.resume_path)
        # Verify the directory exists
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experiment directory not found: {exp_dir}")
    else:
        # Create new experiment directory
        base_dir = cfg.base_dir
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

    best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")
    log_path = os.path.join(exp_dir, "log.txt")

    # Create model
    model = create_model()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    #  Scheduler: Warmup + ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', # mode='min': riduce quando la metrica scende (loss o mpjpe)
        factor=0.5, # fdimezza il LR quando scatta (moltiplica per 0.5)
        patience=cfg.patience // 2, # aspetta n epoche senza miglioramenti prima di ridurre, solitamente metà della cfg.patience dell'early stopping
        verbose=False,
        min_lr=1e-6
    )
    # 1. Scheduler per il Warmup: sale linearmente per N epoche
    #warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    #    optimizer, 
    #    start_factor=0.1, # Parte dal 10% del LR impostato
    #    end_factor=1.0, 
    #    total_iters=cfg.warmup_epochs
    #)
#
    ## 2. Scheduler per il Coseno: parte dopo il warmup
    #cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, 
    #    T_max=cfg.epochs - cfg.warmup_epochs, 
    #    eta_min=1e-6
    #)
#
    ## 3. Concatenazione: SequentialLR gestisce lo switch automaticamente
    #scheduler = torch.optim.lr_scheduler.SequentialLR(
    #    optimizer, 
    #    schedulers=[warmup_scheduler, cosine_scheduler], 
    #    milestones=[cfg.warmup_epochs]
    #)

    # In pratica:
    #    AdamW aggiorna i pesi.
    #    Warmup evita di “spaventare” la rete con un LR troppo alto all’inizio.

    start_epoch = 0
    patience_counter = 0
    current_lr = cfg.lr
    best_val = float("inf")

    # === Resume training se richiesto ===
    if args.resume and args.resume_path is not None and os.path.isfile(args.resume_path):
        print(f"Riprendo il training da {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=cfg.device)
        start_epoch = checkpoint["epoch"] + 1
        patience_counter = checkpoint["patience_counter"]

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #scheduler.load_state_dict(checkpoint["scheduler"])

        best_val = checkpoint.get("val_mpjpe", float("inf"))
        print(f"Ripreso da epoch {start_epoch}, best val MPJPE = {best_val:.2f}")

    model = model.to(device)

    # === Setup logging ===
    # ⚡️ Se siamo in resume → appende. Altrimenti crea nuovo log.
    log_mode = "a" if (args.resume and args.resume_path is not None) else "w"

    with open(log_path, log_mode) as f:
        if log_mode == "w":  # nuovo training → scrivi intestazione
            f.write("==== Config ====\n")
            for k, v in cfg.__dict__.items():
                f.write(f"{k}: {v}\n")

            f.write("\n==== Model ====\n")
            summary_str = str(summary(
                model,
                input_size=(cfg.batch_size, cfg.clip_len, cfg.num_joints, 3),
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=3,
                device=device,
                verbose=0,
            ))
            f.write(summary_str + "\n")
            f.write("\n==== Training Logs ====\n")

        else:
            # append: segna in log che il training è stato ripreso
            f.write("\n==== Resume Training ====\n")
            f.write(f"Start epoch: {start_epoch}\n")


    # Training
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{cfg.epochs}")

        # --- Gestione Warmup ---
        is_warmup = adjust_learning_rate_warmup(optimizer, epoch, cfg.warmup_epochs, cfg.lr)
            
        # --- Training ---
        loss = train_one_epoch(model, train_loader, optimizer, cfg)

        # --- Validation ---
        va = evaluate(model, val_loader, cfg)

        # --- Step dello Scheduler ---
        if not is_warmup:
            scheduler.step(va['mpjpe'])
            #scheduler.step()
            #current_lr = decay_lr_exponentially(current_lr, optimizer)
            current_lr = optimizer.param_groups[0]['lr']  
        else:
            current_lr = optimizer.param_groups[0]['lr']    # just for logging

        # === Salvataggio di sicurezza ===
        epoch_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch+1}.pth")
        torch.save({
            'cfg': cfg.__dict__,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            #'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'val_mpjpe': best_val,
            'patience_counter': patience_counter,
        }, epoch_ckpt_path)
        print(f"Checkpoint salvato: {epoch_ckpt_path}")

        # 🔹 Elimina il file di due epoche fa
        old_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch-1}.pth")
        if os.path.exists(old_ckpt_path):
            os.remove(old_ckpt_path)
            print(f"Checkpoint vecchio rimosso: {old_ckpt_path}")

        epoch_time_hours = (time.time() - epoch_start) / 3600.0

        log_line = (
            f"Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"LR {current_lr:.9f} | "
            f"train mpjpe {loss['mpjpe'] * 1000:.4f} mm | "
            f"train bone {loss['bone'] * 1000:.4f} mm | "
            f"train velocity {loss['velocity'] * 1000:.4f} mm/f^2 | "
            f"train acceleration {loss['acceleration'] * 1000:.4f} mm/f^3 | "
            f"val mpjpe {va['mpjpe'] * 1000:.4f} mm | "
            f"val p_mpjpe {va['p_mpjpe'] * 1000:.4f} mm | "
            f"time {epoch_time_hours * 60:.2f}min"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # save best
        if va['mpjpe'] < best_val:
            patience_counter = 0
            best_val = va['mpjpe']
            torch.save({
                'cfg': cfg.__dict__,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                #'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_mpjpe': best_val,
                'patience_counter': patience_counter,
            }, best_ckpt_path)
            print(f"Saved best checkpoint (val MPJPE={best_val:.2f})")

        patience_counter += 1
        if patience_counter > cfg.patience:
            early = (f"Early stopping at epoch {epoch+1} due to no improvement in {cfg.patience} epochs. The best number of epochs is {epoch+1 - cfg.patience}.")
            print(early)
            with open(log_path, "a") as f:
                f.write(early + "\n")
            break
            
    return best_val

if __name__ == "__main__":
    main()