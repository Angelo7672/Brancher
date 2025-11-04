import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from datetime import datetime
import time
import argparse
import numpy as np

from model import create_model
from config import Config
from sportspose.dataset import SportsPoseDataset


# ------------------------
# Loss functions
# ------------------------
def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_p_mpjpe(predicted, target):
    """
    Protocol #2 MPJPE (rigid alignment) per batch di sequenze (B,T,J,3)
    """
    assert predicted.shape == target.shape, \
        f"Shape mismatch: {predicted.shape} vs {target.shape}"

    B, T, J, D = predicted.shape
    predicted = predicted.reshape(B*T, J, D)
    target = target.reshape(B*T, J, D)

    muX = target.mean(1, keepdim=True)
    muY = predicted.mean(1, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY

    X0 /= X0.norm(dim=(1,2), keepdim=True)
    Y0 /= Y0.norm(dim=(1,2), keepdim=True)

    H = X0.transpose(1,2) @ Y0
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1,2)
    R = V @ U.transpose(1,2)

    detR = torch.linalg.det(R)
    neg_mask = detR < 0
    if neg_mask.any():
        V[neg_mask, :, -1] *= -1
        R[neg_mask] = V[neg_mask] @ U[neg_mask].transpose(1,2)

    scale = (s.sum(1, keepdim=True).unsqueeze(2) *
             X0.norm(dim=(1,2), keepdim=True) /
             Y0.norm(dim=(1,2), keepdim=True))

    t = muX - scale * (muY @ R)
    aligned = scale * (predicted @ R) + t

    return torch.mean(torch.norm(aligned - target, dim=-1))


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


def resize_joints_2d(joints, orig_h, orig_w, target_size):
    target_h, target_w = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    joints_resized = joints.clone()
    joints_resized[..., 0] *= scale_x  # scala x
    joints_resized[..., 1] *= scale_y  # scala y

    return joints_resized


def collate_and_resize(batch, target_size=(256,192)):
    new_batch = {}

    # video
    new_batch["video"] = {"image": {}}
    for view in batch[0]["video"]["image"]:
        vids = []
        for b in batch:
            v = b["video"]["image"][view]
            v = torch.from_numpy(v).permute(0, 3, 1, 2)  # (T, C, H, W)
            v = F.interpolate(v, size=target_size, mode="bilinear", align_corners=False)  # (T, C, H, W)
            vids.append(v)
        
        vids = torch.stack(vids)  # (B, T, C, target, target)
        new_batch["video"]["image"][view] = vids.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)

    # joints_3d
    gts = [b["joints_3d"]["data_points"] for b in batch]
    gts = torch.stack([torch.from_numpy(g).float() for g in gts])
    new_batch["joints_3d"] = {"data_points": gts}

    # joints_2d
    j2ds = {}
    for view in batch[0]["joints_2d"]:
        arrs = []
        for b in batch:
            j = b["joints_2d"][view]
            if isinstance(j, np.ndarray):
                j = torch.from_numpy(j).float()

            H, W = b["video"]["image"][view].shape[1:3]  # altezza, larghezza originali
            j = resize_joints_2d(j, H, W, target_size)
            arrs.append(j)
        j2ds[view] = torch.stack(arrs)
    new_batch["joints_2d"] = j2ds

    return new_batch


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

        # move joints to device
        gt = batch["joints_3d"]["data_points"].to(device)
        batch_size = gt.size(0)

        # batch["video"]["image"] = {
        #     "front": (B, T, H, W, C),
        #     "left": (B, T, H, W, C),
        #     "right": (B, T, H, W, C),
        #     ...
        # }

        # --- Ottieni le viste disponibili ---
        views = list(batch["video"]["image"].keys())
        n_views = len(views)

        # --- Costruisci un batch unico con tutte le viste ---
        videos = [batch["video"]["image"][v] for v in views]  # lista di (B, T, H, W, C)
        video_all = torch.cat(videos, dim=0).to(device)       # (B*V, T, H, W, C)

        # --- Espandi il ground truth per ciascuna vista ---
        gt_expanded = gt.repeat_interleave(n_views, dim=0)    # (B*V, T, J, 3)

        # --- Azzera i gradienti ---
        optimizer.zero_grad()

        # --- Forward pass unico (usa AMP se abilitato) ---
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(video_all)  # (B*V, T, J, 3)

            # --- Root-relative handling ---
            # pelvis = average of joints 11 and 12
            pelvis = gt_expanded[:, :, [11, 12], :].mean(dim=2, keepdim=True)  # (B, T, 1, 3)
            gt_rel = gt_expanded - pelvis   # Place the depth of first frame root to 0.

            # --- Compute losses ---
            loss = loss_mpjpe(pred, gt_rel)

        # --- Backward + step ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- Update dei meter ---
        loss_meter.update(loss.item(), batch_size * n_views)


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

            # --- Ground truth joints ---
            gt = batch["joints_3d"]["data_points"].to(device)
            batch_size = gt.size(0)

            # --- Ottieni le viste disponibili ---
            views = list(batch["video"]["image"].keys())
            n_views = len(views)

            # --- Costruisci un batch unico con tutte le viste ---
            videos = [batch["video"]["image"][v] for v in views]  # lista di (B, T, H, W, C)
            video_all = torch.cat(videos, dim=0).to(device)       # (B*V, T, H, W, C)

            # --- Espandi il ground truth per ciascuna vista ---
            gt_expanded = gt.repeat_interleave(n_views, dim=0)    # (B*V, T, J, 3)

            # --- Forward ---
            pred = model(video_all)  # (B*V, T, J, 3)

            # --- Root-relative using pelvis ---
            pelvis_pred = pred[:, :, [11, 12], :].mean(dim=2, keepdim=True)
            pelvis_gt   = gt_expanded[:, :, [11, 12], :].mean(dim=2, keepdim=True)

            pred_rel = pred - pelvis_pred
            gt_rel = gt_expanded - pelvis_gt

            err1 = loss_mpjpe(pred_rel, gt_rel)
            err2 = loss_p_mpjpe(pred_rel, gt_rel)

            mpjpe_meter.update(err1.item(), batch_size)
            p_mpjpe_meter.update(err2.item(), batch_size)

    results = {"mpjpe": mpjpe_meter.avg, "p_mpjpe": p_mpjpe_meter.avg}
    return results


# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=None, help="Resume training from this epoch")
    
    args = parser.parse_args()

    cfg = Config()
    device = cfg.device

    # Create or get experiment directory
    if args.resume and args.resume_path:
        # Extract experiment directory from checkpoint path by removing the weight filename
        exp_dir = os.path.dirname(args.resume_path)
        # Verify the directory exists
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experiment directory not found: {exp_dir}")
    else:
        # Create new experiment directory
        base_dir = "experiments/baseline_2d_lift"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

    best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")
    log_path = os.path.join(exp_dir, "log.txt")

    model = create_model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # AMP scaler, Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    start_epoch = 0
    best_val = float("inf")

    # === Resume training se richiesto ===
    if args.resume and args.resume_path is not None and os.path.isfile(args.resume_path):
        print(f"Riprendo il training da {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        if args.start_epoch is None:
            start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = args.start_epoch  # override manuale

        best_val = checkpoint.get("val_mpjpe", float("inf"))
        print(f"Ripreso da epoch {start_epoch}, best val MPJPE = {best_val:.2f}")

    model = model.to(device)


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
                input_size=(cfg.batch_size, cfg.clip_len, cfg.frame_size[0], cfg.frame_size[1], 3),
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
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


    print("Impostazione dataset e dataloader...")

    # Datasets
    train_ds = SportsPoseDataset(
        data_dir=cfg.datapath,
        sample_level="video",
        seq_size=cfg.clip_len,
        blacklist={
            "metadata": {"activity": ["throw_baseball"]}
        }
    )
    val_ds = SportsPoseDataset(
        data_dir=cfg.datapath,
        sample_level="video",
        seq_size=cfg.clip_len,
        validation_dataset=True,
        whitelist={
            "metadata": {"activity": ["throw_baseball"]}
        }
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True, # drop_last per evitare batch più piccoli alla fine
        collate_fn=lambda x: collate_and_resize(x, target_size=cfg.frame_size),
    ) 
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_and_resize(x, target_size=cfg.frame_size),
    )

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

    # Training
    total_epochs = cfg.epochs

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{total_epochs}")

        losses = AverageMeter()

        train_one_epoch(model, train_loader, optimizer, scaler, losses, cfg)

        # === Salvataggio di sicurezza prima della valutazione ===
        epoch_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch+1}.pth")
        torch.save({
            'cfg': cfg.__dict__,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_mpjpe': best_val,  # ancora non aggiornato, ma utile per resume
        }, epoch_ckpt_path)
        print(f"Checkpoint salvato: {epoch_ckpt_path}")

        # 🔹 Elimina il file di due epoche fa
        old_ckpt_path = os.path.join(exp_dir, f"weight_epoch{epoch-1}.pth")
        if os.path.exists(old_ckpt_path):
            os.remove(old_ckpt_path)
            print(f"Checkpoint vecchio rimosso: {old_ckpt_path}")

        va = evaluate(model, val_loader, cfg)
        scheduler.step()  # step del LR scheduler

        epoch_time_hours = (time.time() - epoch_start) / 3600.0
        current_lr = optimizer.param_groups[0]['lr']

        log_line = (
            f"Epoch {epoch+1:03d}/{total_epochs} | "
            f"LR {current_lr:.6f} | "
            f"train loss {losses.avg:.4f} | "
            f"val mpjpe {va['mpjpe']:.2f} | "
            f"val p_mpjpe {va['p_mpjpe']:.2f} | "
            f"time {epoch_time_hours:.3f}h"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # save best
        if va['mpjpe'] < best_val:
            best_val = va['mpjpe']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'cfg': cfg.__dict__,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_mpjpe': best_val,
            }, best_ckpt_path)
            print(f"Saved best checkpoint (val MPJPE={best_val:.2f})")

if __name__ == "__main__":
    main()