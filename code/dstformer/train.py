import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from datetime import datetime
import argparse

from model import create_model
from config import Config
from sportspose.dataset import SportsPoseDataset

import numpy as np
import cv2

# ------------------------
# Losses & Metrics
# ------------------------
def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Per Joint Position Error (mm). pred, gt: (B, T, J, 3)."""
    """Risultato: errore medio di posizione per joint."""
    assert pred.shape == gt.shape
    return torch.mean(torch.linalg.norm(pred - gt, dim=-1)) # differenza vettoriale fra posizione predetta e ground truth.


def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor, bones) -> torch.Tensor:
    """L1 loss on bone lengths to encourage realistic limb proportions."""
    """Qui si vuole che il modello rispetti le proporzioni anatomiche (le ossa non cambiano lunghezza tra frame)."""
    def bone_lengths(x):
        """Calcola la distanza Euclidea tra ogni coppia di joints connessi da un osso."""
        """torch.stack → costruisce un tensore (B, T, Bn) con tutte le lunghezze ossee, dove Bn = numero di ossa."""
        diffs = []
        for i, p in bones: # bones → lista di coppie (i, p) che indicano quali joints sono collegati (ad esempio, spalla–gomito, gomito–polso).
            diffs.append(torch.linalg.norm(x[..., i, :] - x[..., p, :], dim=-1))  # (B, T)
        return torch.stack(diffs, dim=-1)  # (B, T, Bn)

    bl_pred = bone_lengths(pred)    # lunghezze ossee predette.
    bl_gt = bone_lengths(gt)    # lunghezze ossee ground truth.
    return F.l1_loss(bl_pred, bl_gt)    # penalizza deviazioni con una L1 loss (più robusta agli outlier rispetto all’MSE).


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


# -------------------------
# Training / Evaluation
# -------------------------
class AverageMeter:
    """AverageMeter tiene traccia della media progressiva di una metrica durante il training/valutazione."""
    def __init__(self):
        self.reset()    # Alla creazione inizializza i contatori.
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val: float, n: int = 1): # aggiorna con un nuovo valore val, pesato per n (es. batch size).
        # Questo permette di fare una media pesata: se un batch ha più esempi, contribuisce proporzionalmente di più.
        self.sum += val * n
        self.cnt += n
    @property
    def avg(self): # calcola la media pesata.
        return self.sum / max(1, self.cnt)


def train_one_epoch(model, loader, optimizer, scaler, cfg: Config):
    model.train()   # set model to training mode
    loss_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    
    print("I'm trainining")
    i = 0

    # Lista delle viste disponibili
    views = ['front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left']

    for batch in loader:
        print(f" Batch {i+1}/{len(loader)}", end='\r')
        i += 1
        # Ground truth 3D pose (B, T, J, 3)
        gt = batch["joints_3d"]["data_points"]
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float()
        gt = gt.to(cfg.device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            # Itera sulle viste senza concatenare
            for view in views:
                video = batch["video"]["image"][view]  # (B, T, H, W, 3)
                if isinstance(video, np.ndarray):
                    video = torch.from_numpy(video).float()
                video = video.to(cfg.device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    pred = model(video)  # (B, T, J, 3)

                    # Loss
                    loss_pos = F.mse_loss(pred, gt)
                    loss_bone = bone_length_loss(pred, gt, cfg.bones) * cfg.lambda_bone
                    loss = loss_pos #+ loss_bone

                    err = mpjpe(pred, gt)

                # Backward immediato
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

                # Logging
                bs = gt.size(0)
                loss_meter.update(loss.item(), bs)
                mpjpe_meter.update(err.item(), bs)

    return {"loss": loss_meter.avg, "mpjpe": mpjpe_meter.avg}


@torch.no_grad()
def evaluate(model, loader, cfg: Config):
    model.eval()
    mpjpe_meter = AverageMeter()

    print("I'm evaluating")
    i = 0

    # Lista delle viste disponibili
    views = ['front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left']

    for batch in loader:
        print(f" Batch {i+1}/{len(loader)}", end='\r')
        i += 1
        # Ground truth 3D pose (B, T, J, 3)
        gt = batch["joints_3d"]["data_points"]
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float()
        gt = gt.to(cfg.device, dtype=torch.float32)

        # Itera sulle viste senza concatenare
        for view in views:
            video = batch["video"]["image"][view]  # (B, T, H, W, 3)
            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video).float()
            video = video.to(cfg.device, dtype=torch.float32)

            # Predizione
            pred = model(video)

            # Metrica MPJP
            err = mpjpe(pred, gt)

            # Aggiorna il meter usando la dimensione reale del batch (non ×7)
            bs = gt.size(0)
            mpjpe_meter.update(err.item(), bs)

    return {"mpjpe": mpjpe_meter.avg}


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

    # Create or get experiment directory
    if args.resume and args.resume_path:
        # Extract experiment directory from checkpoint path by removing the weight filename
        exp_dir = os.path.dirname(args.resume_path)
        # Verify the directory exists
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experiment directory not found: {exp_dir}")
    else:
        # Create new experiment directory
        base_dir = "/Experiments/angeloa00/dstformer"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

    best_ckpt_path = os.path.join(exp_dir, "best_weight.pth")
    log_path = os.path.join(exp_dir, "log.txt")

    # Costruiamo il modello
    model = create_model()

    # Optimizer
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
            input_size=(cfg.batch_size, cfg.clip_len, cfg.img_size[0], cfg.img_size[1], 3),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=cfg.device,
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
            "metadata": {
                "activity": ["throw_baseball", "volley"]
            }
        }
    )
    val_ds = SportsPoseDataset(
        data_dir=cfg.datapath,
        sample_level="video",
        seq_size=cfg.clip_len,
        validation_dataset=True,
        whitelist={
            "metadata": {
                "activity": ["throw_baseball"]
            }
        }
    )
    test_ds = SportsPoseDataset(
        data_dir=cfg.datapath,
        sample_level="video",
        seq_size=cfg.clip_len,
        whitelist={
            "metadata": {
                "activity": ["volley"]
            }
        }
    )

    # DataLoaders con configurazione ottimizzata per memoria
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True, # drop_last per evitare batch più piccoli alla fine
        #persistent_workers=False,  # ← False per risparmiare memoria
        collate_fn=lambda x: collate_and_resize(x, target_size=cfg.img_size),
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        #persistent_workers=False,  # ← False per risparmiare memoria
        collate_fn=lambda x: collate_and_resize(x, target_size=cfg.img_size),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,  
        shuffle=False,
        num_workers=cfg.num_workers, 
        pin_memory=True,
        #persistent_workers=False,  # ← False per risparmiare memoria
        collate_fn=lambda x: collate_and_resize(x, target_size=cfg.img_size),
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
    remaining_epochs = total_epochs - start_epoch

    with open(log_path, "a") as f:
        f.write("\n==== Training Logs ====\n")

    for epoch in range(start_epoch, total_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)

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
        scheduler.step()

        log_line = f"Epoch {epoch+1:03d}/{total_epochs} | train loss {tr['loss']:.4f} mpjpe {tr['mpjpe']:.2f} | val mpjpe {va['mpjpe']:.2f}"
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

     # === Valutazione finale sul test set ===
    print("\nValutazione finale sul test set...")

    # Ricarichiamo il best checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model'])

    test_metrics = evaluate(model, test_loader, cfg)
    print(f"Test MPJPE: {test_metrics['mpjpe']:.2f}")

    # Salviamo anche sul log
    with open(log_path, "a") as f:
        f.write("\n==== Test Results ====\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()