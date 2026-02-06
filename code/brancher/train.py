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

from model import create_model
from config import Config
from loss import *
from rotations import *
from learning import *


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
        pose2d_path = os.path.join(sample_dir, "pose2d.json")
        heatmap_path = os.path.join(sample_dir, "heatmaps.pt")
        pose3d_path = os.path.join(sample_dir, "pose3d.json")

        # load 2D poses
        with open(pose2d_path, 'r') as f2:
            j2d = torch.tensor(json.load(f2), dtype=torch.float32)

        # Load heatmaps
        heatmaps = torch.load(heatmap_path, map_location='cpu')

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
            "heatmaps": heatmaps,   # (T, J, H, W)
            "joints_3d": j3d     # (T, J, 3)
        }


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


def train_one_epoch(model, loader, optimizer, loss_wrapper, cfg: Config):
    device = cfg.device
    model.train()

    print("I'm trainining")
    batch_idx = 0

    mpjpe_meter = AverageMeter()
    bone_meter = AverageMeter()
    velocity_meter = AverageMeter()
    acceleration_meter = AverageMeter()
    hm_meter = AverageMeter()
    geodesic_meter = AverageMeter()
    weight_meters = {k: AverageMeter() for k in ['mpjpe', 'bone', 'velocity', 'acceleration', 'hm']}#, 'geodesic', 'hm']}

    for batch in loader:    
        print(f" Batch {batch_idx+1}/{len(loader)}", end='\r')
        batch_idx += 1

        # {
        #     "element": [
        #         {
        #             "video": tensor(T, H, W, C),
        #             "joints_2d": tensor(T, J, 3),
        #             "joints_3d": tensor(T, J, 3),
        #             "heatmaps": tensor(T, J, H, W),
        #             "metadata": {...}
        #         }
        #     ]
        # }

        gt = batch["joints_3d"].to(device)
        video = batch["joints_2d"].to(device)  # (B, T, J, 3)
        #heatmap = batch["heatmaps"].to(device)    # (B, T, J, H, W)

        optimizer.zero_grad()

        # --- Forward ---
        pos, rot, htmp = model(video)  # (B, T, J, 3), (B, T, bones, 3, 3), (B, T, J, H, W)
        
        # --- Root-relative handling ---
        root_gt = gt[:, :, [12], :]
        gt_rel = (gt - root_gt) / cfg.scale_factor  # normalizza a [-1, 1] per Tanh

        # --- Compute losses ---

        # Rotation Branch
        t_pose = get_t_pose_coco(device)  # [13, 3]
        bones = get_coco_bones(pos)  # [B, T, 13, 3]
        r_p = get_rotation_between_vectors(t_pose, bones)  # [B, T, 13, 3, 3]

        # Total Loss
        losses = {
            'mpjpe': loss_mpjpe(pos, gt_rel),
            'bone': loss_limb_var(pos),
            'velocity': velocity_smoothness_loss(pos),
            'acceleration': acceleration_smoothness_loss(pos),
            #'geodesic': geodesic_loss(rot, r_p),
            'hm': heatmap_loss(htmp, heatmap)
        }

        # --- Bilanciamento automatico ---
        total_loss = loss_wrapper(losses)

        # --- Backward + step ---
        total_loss.backward()
        optimizer.step()

        mpjpe_meter.update(losses['mpjpe'].item(), gt.size(0))
        bone_meter.update(losses['bone'].item(), gt.size(0))
        velocity_meter.update(losses['velocity'].item(), gt.size(0))
        acceleration_meter.update(losses['acceleration'].item(), gt.size(0))
        #geodesic_meter.update(losses['geodesic'].item(), gt.size(0))
        hm_meter.update(losses['hm'].item(), heatmap.size(0))

        # Log dei pesi correnti
        with torch.no_grad():
            # Il peso effettivo che moltiplica la loss è exp(-s)
            weights = torch.exp(-loss_wrapper.log_vars)
            keys = ['mpjpe', 'bone', 'velocity', 'acceleration', 'hm']#, 'geodesic', 'hm']
            for i, k in enumerate(keys):
                weight_meters[k].update(weights[i].item())
    

    results = {"mpjpe": mpjpe_meter.avg, "bone": bone_meter.avg, "velocity": velocity_meter.avg, "acceleration": acceleration_meter.avg, "heatmap": hm_meter.avg,
                "weights": {k: v.avg for k, v in weight_meters.items()}}
               #"geodesic": geodesic_meter.avg, "heatmap": hm_meter.avg, "weights": {k: v.avg for k, v in weight_meters.items()}}
    
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
            video = batch["joints_2d"].to(device)  # (B, T, J, 3)
            #heatmap = batch["heatmaps"].to(device)    # (B, T, J, H, W)

            # --- Forward ---
            pred,_,_ = model(video)  # (B, T, J, 3), (B, T, bones, 3, 3), (B, T, J, H, W)

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
    parser.add_argument("--finetune", action="store_true", help="Fine-tune from pretrained weights")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()

    cfg = Config()
    device = cfg.device

    print("Setting up the dataset and dataloader...")
    if args.finetune:
        dataset_name = "air"
        data_dir = cfg.datapath_air
        frame_size = cfg.frame_size_air
    else:
        dataset_name = "sports"
        data_dir = cfg.datapath_sport
        frame_size = cfg.frame_size_sport

    # Datasets
    train_loader, val_loader = create_dataloaders(
        dataset_name,
        data_dir,
        batch_size=cfg.batch_size,
        frame_size=frame_size,
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
        if args.finetune:
            best_ckpt_path = os.path.join(exp_dir, "best_weight_ft.pth")
        else:
            best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")
    elif args.finetune:
        exp_dir = os.path.dirname(cfg.pretrained_weights)
        best_ckpt_path = os.path.join(exp_dir, "best_weight_ft.pth")
    else:
        # Create new experiment directory
        base_dir = cfg.base_dir
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")

    log_path = os.path.join(exp_dir, "log.txt")

    # Create model
    model = create_model(frame_size)
    if args.finetune:
        print("Fine-tuning from pretrained weights...")
        model = load_pretrained_weights(model, torch.load(cfg.pretrained_weights, map_location=cfg.device))
        

    # Wrapper delle loss
    loss_wrapper = MultiTaskLossWrapper(num_tasks=6).to(device)
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_wrapper.parameters()), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )


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
        loss_wrapper.load_state_dict(checkpoint["loss_wrapper"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        best_val = checkpoint.get("val_mpjpe", float("inf"))
        print(f"Ripreso da epoch {start_epoch}, best val MPJPE = {best_val:.2f}")

    model = model.to(device)

    # === Setup logging ===
    # ⚡️ If we're in resume mode, append. Otherwise, create a new log.
    log_mode = "a" if (args.resume and args.resume_path is not None) else "w"

    if args.finetune:
        with open(log_path, "a") as f:
            if args.resume:
                f.write("\n==== Resume Finetuning ====\n")
                f.write(f"Start epoch: {start_epoch}\n")
            else:
                f.write("\n==== Finetuning ====\n")
    else:
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
                    device=device,
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

        # --- Gestione Warmup ---
        is_warmup = adjust_learning_rate_warmup(optimizer, epoch, cfg.warmup_epochs, cfg.lr)
            
        # --- Training ---
        loss = train_one_epoch(model, train_loader, optimizer, loss_wrapper, cfg)

        # --- Validation ---
        va = evaluate(model, val_loader, cfg)

        # --- Step dello Scheduler ---
        if not is_warmup:
            current_lr = decay_lr_exponentially(current_lr, optimizer)

        current_lr = optimizer.param_groups[0]['lr']    # just for logging

        # === Safety rescue ===
        if args.finetune:
            epoch_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch+1}_ft.pth")
        else:
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
        if args.finetune:
            old_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch-1}_ft.pth")
        else:
            old_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch-1}.pth")
        if os.path.exists(old_ckpt_path):
            os.remove(old_ckpt_path)
            print(f"Old checkpoint removed: {old_ckpt_path}")

        epoch_time_hours = (time.time() - epoch_start) / 3600.0

        log_line = (
            f"Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"LR {current_lr:.9f} | "
            f"train mpjpe {loss['mpjpe'] * 1000:.4f} mm | "
            f"train bone {loss['bone'] * 1000:.4f} mm | "
            f"train velocity {loss['velocity'] * 1000:.4f} mm/f^2 | "
            f"train acceleration {loss['acceleration'] * 1000:.4f} mm/f^3 | "
            #f"train geodesic {loss['geodesic'] * (180/math.pi):.4f} deg | "
            f"train hm_loss {loss['heatmap']:.7f} | "
            "\n"
            f"W_mpjpe {loss['weights']['mpjpe']:.4f} | "
            f"W_bone {loss['weights']['bone']:.4f} | "
            f"W_velocity {loss['weights']['velocity']:.4f} | "
            f"W_acceleration {loss['weights']['acceleration']:.4f} | "
            #f"W_geodesic {loss['weights']['geodesic']:.4f} | "
            f"W_heatmap {loss['weights']['hm']:.4f} | "
            "\n"
            f"val mpjpe {va['mpjpe'] * 1000:.4f} mm | "
            f"val p_mpjpe {va['p_mpjpe'] * 1000:.4f} mm | "
            f"Elapsed time {epoch_time_hours * 60:.2f}min"
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
                'loss_wrapper': loss_wrapper.state_dict(),
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