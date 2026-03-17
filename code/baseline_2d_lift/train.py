import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pickle
from torchinfo import summary
from datetime import datetime
import time
import argparse

from model import create_model
from config import Config
from loss import *
from learning import *

# ------------------------
# DataLoader
# ------------------------
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
        
        sample = {
            "joints_2d": j2d,
            "joints_3d": j3d,
        }

        return sample


def create_dataloaders(data_dir, batch_size, num_workers, seed):
    print("Using AthletePose dataset")
    train_ds = AthletePosePreprocessed(os.path.join(data_dir, "train"))
    val_ds = AthletePosePreprocessed(os.path.join(data_dir, "valid"))

    train_len = len(train_ds)
    val_len = len(val_ds)

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

    print(f"Train: {train_len} | Val: {val_len}")
    return train_loader, val_loader


def adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, target_lr):
    if epoch < warmup_epochs:
        lr = target_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return True # Active warmup
    return False # Warmup finished


def decay_lr_exponentially(lr, optimizer, lr_decay=0.99):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr


# ------------------------
# Training
# ------------------------
def train_one_epoch(model, loader, optimizer, cfg: Config):
    device = cfg.device
    model.train()

    print("I'm trainining")
    batch_idx = 0

    mpjpe_meter = AverageMeter()

    for batch in loader:    
        print(f" Batch {batch_idx+1}/{len(loader)}", end='\r')
        batch_idx += 1

        gt = batch["joints_3d"].to(device)
        video = batch["joints_2d"].to(device)  # (B, T, J, 3)

        optimizer.zero_grad()   # Reset the gradients

        pred = model(video)  # (B, T, J, 3)
        
        # --- Root-relative handling ---
        root_gt = gt[:, :, [0], :]
        gt_rel = gt - root_gt
        
        # --- Compute losses ---
        loss = loss_mpjpe(pred, gt_rel)

        mpjpe_meter.update(loss.item(), gt.size(0))

        # --- Backward + step ---
        loss.backward()
        optimizer.step()

    results = {"mpjpe": mpjpe_meter.avg}
    
    return results


# ------------------------
# Validation
# ------------------------
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
            video = batch["joints_2d"].to(device)  # (B, T, J, 3)

            # --- Forward ---
            pred= model(video)   # (B, T, J, 3)

            # --- Root-relative ---
            root_pred = pred[:, :, [0], :]
            root_gt   = gt[:, :, [0], :]

            pred_rel = pred - root_pred
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

    print("Setting up the dataset and dataloader...")

    # Datasets
    train_loader, val_loader = create_dataloaders(
        cfg.datapath_athlete,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Create or get experiment directory
    if args.resume and args.resume_path:
        # Extract experiment directory from checkpoint path by removing the weight filename
        exp_dir = os.path.dirname(args.resume_path)
        # Verify the directory exists
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experiment directory not found: {exp_dir}")
        best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")
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
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    # === Training variables ===
    start_epoch = 0
    patience_counter = 0
    current_lr = cfg.lr
    best_val = float("inf")

    # === Resume training if required ===
    if args.resume and args.resume_path is not None and os.path.isfile(args.resume_path):
        print(f"I'm resuming training from {args.resume_path}")
       
        checkpoint = torch.load(args.resume_path, map_location='cpu')
        start_epoch = checkpoint["epoch"] + 1
        patience_counter = checkpoint["patience_counter"]

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        best_val = checkpoint.get("val_mpjpe", float("inf"))

        del checkpoint
        torch.cuda.empty_cache()
        
        print(f"Resumed training from epoch {start_epoch}, best val MPJPE = {best_val:.2f}")

    model = model.to(cfg.device)

    # === Setup logging ===
    # ⚡️ If we're in resume mode, append. Otherwise, create a new log.
    log_mode = "a" if (args.resume and args.resume_path is not None) else "w"

    with open(log_path, log_mode) as f:
        if log_mode == "w":
            f.write("==== Config ====\n")
            for k, v in cfg.__dict__.items():
                f.write(f"{k}: {v}\n")

            f.write("\n==== Model ====\n")
            summary_str = str(summary(
                model,
                input_size=(cfg.batch_size, cfg.clip_len, cfg.num_joints, 3),
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=3,
                device=cfg.device,
                verbose=0,
            ))
            f.write(summary_str + "\n")
            f.write("\n==== Training Logs ====\n")
        else:
            f.write("\n==== Resume Training ====\n")
            f.write(f"Start epoch: {start_epoch}\n")


    # Training
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{cfg.epochs}")

        # --- Warmup Management ---
        is_warmup = adjust_learning_rate_warmup(optimizer, epoch, cfg.warmup_epochs, cfg.lr)
            
        # --- Training ---
        loss = train_one_epoch(model, train_loader, optimizer, cfg)

        # --- Validation ---
        va = evaluate(model, val_loader, cfg)

        # --- Step of the Scheduler ---
        if not is_warmup:
            current_lr = decay_lr_exponentially(current_lr, optimizer)

        current_lr = optimizer.param_groups[0]['lr']    # just for logging

        # === Safety rescue ===
        epoch_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch+1}.pth")
        torch.save({
            'cfg': cfg.__dict__,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_mpjpe': best_val,
            'patience_counter': patience_counter,
        }, epoch_ckpt_path)
        print(f"Checkpoint saved: {epoch_ckpt_path}")

        # 🔹 Delete the file from two epochs ago
        old_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch-1}.pth")
        if os.path.exists(old_ckpt_path):
            os.remove(old_ckpt_path)
            print(f"Old checkpoint removed: {old_ckpt_path}")

        epoch_time_hours = (time.time() - epoch_start) / 3600.0
        log_line = (
            f"Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"LR {current_lr:.9f} | "
            f"train mpjpe {loss['mpjpe'] * 1000:.4f} mm | "
            f"val mpjpe {va['mpjpe'] * 1000:.4f} mm | "
            f"val p_mpjpe {va['p_mpjpe'] * 1000:.4f} mm | "
            f"time {epoch_time_hours * 60:.2f}min"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # === Save best ===
        if va['mpjpe'] < best_val:
            patience_counter = 0
            best_val = va['mpjpe']
            torch.save({
                'cfg': cfg.__dict__,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
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