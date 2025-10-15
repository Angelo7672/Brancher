import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from datetime import datetime
import argparse
import numpy as np

from model import create_model
from config import Config
from sportspose.dataset import SportsPoseDataset
from learning import AverageMeter, load_pretrained_weights
from loss import (
    mpjpe,
    p_mpjpe,
    n_mpjpe,
    loss_mpjpe,
    loss_p_mpjpe,
    loss_velocity,
    loss_limb_var,
    loss_limb_gt,
    loss_angle,
    loss_angle_velocity,
    loss_2d_weighted,
)


# ------------------------
# Utility
# ------------------------
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


def train_one_epoch(model, loader, optimizer, scaler, losses, cfg: Config):
    device = cfg.device
    model.train()

    print("I'm trainining")
    i = 0

    # views present in the loader (derived from first batch)
    # but we'll fetch views per-batch to be robust

    for batch in loader:
        print(f" Batch {i+1}/{len(loader)}", end='\r')
        i += 1

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
            if cfg.root_rel == False:
                gt_rel = gt_expanded - gt_expanded[:, :, 0:1, :]
            else:
                gt_rel = gt_expanded.clone()
                gt_rel[..., 2] = gt_rel[..., 2] - gt_rel[:, 0:1, 0:1, 2]

            # --- Compute losses (MotionBERT style) ---
            loss_3d_pos = loss_mpjpe(pred, gt_rel)
            loss_3d_scale = n_mpjpe(pred, gt_rel)
            loss_3d_velocity = loss_velocity(pred, gt_rel)
            #loss_lv = loss_limb_var(pred)
            #loss_lg = loss_limb_gt(pred, gt_rel)
            #loss_a = loss_angle(pred, gt_rel)
            #loss_av = loss_angle_velocity(pred, gt_rel)
            #loss_2d_proj = loss_2d_weighted(pred, gt_rel, None)

            # --- Total loss ---
            loss_total = (
                loss_3d_pos
                + cfg.lambda_scale * loss_3d_scale
                + cfg.lambda_3d_velocity * loss_3d_velocity
                #+ cfg.lambda_lv * loss_lv
                #+ cfg.lambda_lg * loss_lg
                #+ cfg.lambda_a * loss_a
                #+ cfg.lambda_av * loss_av
                #+ cfg.lambda_2d_proj * loss_2d_proj
            )

        # --- Backward + step ---
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- Update dei meter ---
        losses['3d_pos'].update(loss_3d_pos.item(), batch_size * n_views)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size * n_views)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size * n_views)
        #losses['lv'].update(loss_lv.item(), batch_size * n_views)
        #losses['lg'].update(loss_lg.item(), batch_size * n_views)
        #losses['angle'].update(loss_a.item(), batch_size * n_views)
        #losses['angle_velocity'].update(loss_av.item(), batch_size * n_views)
        #losses['2d_proj'].update(loss_2d_proj.item(), batch_size * n_views)
        #losses['total'].update(loss_total.item(), batch_size * n_views)


def evaluate(model, loader, cfg: Config):
    device = cfg.device
    model.eval()

    print("I'm evaluating")
    i = 0

    mpjpe_meter = AverageMeter()
    p_mpjpe_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            print(f" Batch {i+1}/{len(loader)}", end='\r')
            i += 1

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

            # handle root-relative vs absolute for evaluation as in MotionBERT
            if cfg.root_rel==False:
                pred[:, :, 0, :] = 0    # Place the depth of first frame root to 0.
            else:
                gt_expanded[:, 0, 0, 2] = 0

            pred = pred - pred[:, :, 0:1, :]
            gt_expanded = gt_expanded - gt_expanded[:, :, 0:1, :]

            err1 = loss_mpjpe(pred, gt_expanded)
            err2 = loss_p_mpjpe(pred, gt_expanded)

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
        base_dir = "/Experiments/angeloa00/brancher"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

    best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")
    log_path = os.path.join(exp_dir, "log.txt")

    model = create_model(training=True)

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
    else:
        # Carica pesi pre-trained (MotionBERT)
        checkpoint_path = "/Experiments/angeloa00/motionbert/best_epoch.bin"
        model = load_pretrained_weights(model, torch.load(checkpoint_path, map_location=cfg.device))

    model = model.to(device)


    # Apriamo log file
    with open(log_path, "w") as f:
        # Config
        f.write("==== Config ====\n")
        for k, v in cfg.__dict__.items():
            f.write(f"{k}: {v}\n")
        # Parametri e schema
        f.write("\n==== Model ====\n")
        # Salviamo anche schema
        summary_str = str(summary(
            model,
            input_size=(cfg.batch_size, cfg.clip_len, cfg.frame_size[0], cfg.frame_size[1], 3),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=device,
            verbose=0,
        ))
        f.write(summary_str + "\n")

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
    def lr_lambda(it: int) -> float:
        if cfg.warmup_epochs > 0:
            total_steps = len(train_loader) * cfg.warmup_epochs  # batch totali di warmup
            warmup_progress = (it + 1) / total_steps            # da 1/total_steps → 1
            return min(1.0, warmup_progress)                   # clamp a [0,1]
        else:
            return 1.0  # nessun warmup

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) # scheduler che applica lr_lambda ad ogni step.

    # In pratica:
    #    AdamW aggiorna i pesi.
    #    AMP accelera il training con mixed precision.
    #    Warmup evita di “spaventare” la rete con un LR troppo alto all’inizio.

    # Training
    total_epochs = cfg.epochs

    with open(log_path, "a") as f:
        f.write("\n==== Training Logs ====\n")

    for epoch in range(start_epoch, total_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")

        losses = {
            '3d_pos': AverageMeter(),
            '3d_scale': AverageMeter(),
            '2d_proj': AverageMeter(),
            'lg': AverageMeter(),
            'lv': AverageMeter(),
            'total': AverageMeter(),
            '3d_velocity': AverageMeter(),
            'angle': AverageMeter(),
            'angle_velocity': AverageMeter(),
        }

        #train_one_epoch(model, train_loader, optimizer, scaler, losses, cfg)

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
        scheduler.step()  # step del LR scheduler (warmup)

        log_line = f"Epoch {epoch+1:03d}/{total_epochs} | train loss {losses['total'].avg:.4f} | val mpjpe {va['mpjpe']:.2f} | val p_mpjpe {va['p_mpjpe']:.2f}"
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