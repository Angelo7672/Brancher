import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from datetime import datetime

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
    """Risultato: errore medio di posizione per joint, in millimetri (se i dati sono in mm)."""
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
def collate_and_resize(batch, target_size=256):
    new_batch = {}

    # video
    new_batch["video"] = {"image": {}}
    for view in batch[0]["video"]["image"]:
        vids = []
        for b in batch:
            v = b["video"]["image"][view]
            v = torch.from_numpy(v).permute(0, 3, 1, 2) if isinstance(v, np.ndarray) else v.permute(0, 3, 1, 2)  # (T, C, H, W)
            v = F.interpolate(v, size=(target_size, target_size), mode="bilinear", align_corners=False)  # (T, C, target, target)
            vids.append(v)
        
        vids = torch.stack(vids)  # (B, T, C, target, target)
        new_batch["video"]["image"][view] = vids.permute(0, 1, 3, 4, 2)  # (B, T, target, target, C)

    # joints_3d
    gts = [b["joints_3d"]["data_points"] for b in batch]
    gts = torch.stack([
        torch.from_numpy(g).float() if isinstance(g, np.ndarray) else g
        for g in gts
    ])
    new_batch["joints_3d"] = {"data_points": gts}

    # joints_2d
    j2ds = {}
    for view in batch[0]["joints_2d"]:
        arrs = [b["joints_2d"][view] for b in batch]
        arrs = torch.stack([
            torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a
            for a in arrs
        ])
        j2ds[view] = arrs
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
                    loss = cfg.lambda_mpjpe * loss_pos #+ loss_bone

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
    cfg = Config()

    # Creiamo cartella log
    base_dir = "/Experiments/angeloa00/baseline_direct"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(base_dir, f"model_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, "weight.pth")
    log_path = os.path.join(exp_dir, "log.txt")

    # Costruiamo il modello
    model = create_model()

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
            input_size=(cfg.batch_size, cfg.clip_len, cfg.img_size, cfg.img_size, 3),
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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # AMP scaler, Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

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
    best_val = float('inf')
    logs = []

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        va = evaluate(model, val_loader, cfg)
        scheduler.step()

        log_line = f"Epoch {epoch+1:03d}/{cfg.epochs} | train loss {tr['loss']:.4f} mpjpe {tr['mpjpe']:.2f}mm | val mpjpe {va['mpjpe']:.2f}mm"
        print(log_line)
        logs.append(log_line)

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
            }, ckpt_path)
            print(f"Saved best checkpoint (val MPJPE={best_val:.2f}mm)")

    # Scriviamo log completo su file
    with open(log_path, "a") as f:
        f.write("\n==== Training Logs ====\n")
        for line in logs:
            f.write(line + "\n")

     # === Valutazione finale sul test set ===
    print("\nValutazione finale sul test set...")

    # Ricarichiamo il best checkpoint
    checkpoint = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model'])

    test_metrics = evaluate(model, test_loader, cfg)
    print(f"Test MPJPE: {test_metrics['mpjpe']:.2f}mm")

    # Salviamo anche sul log
    with open(log_path, "a") as f:
        f.write("\n==== Test Results ====\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()