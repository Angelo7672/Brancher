import optuna
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob

from config import Config
from loss import *
from learning import AverageMeter
from model import optimize_model


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
        pose3d_path = os.path.join(sample_dir, "pose3d.json")

        # load 2D poses
        with open(pose2d_path, 'r') as f2:
            j2d = torch.tensor(json.load(f2), dtype=torch.float32)

        # load 3D poses
        with open(pose3d_path, 'r') as f3:
            j3d = torch.tensor(json.load(f3), dtype=torch.float32)

        # ---------------------------------------------------------
        # 1. NORMALIZZAZIONE 2D: Da Pixel a [-1, 1]
        # ---------------------------------------------------------
        j2d[..., 0] = (j2d[..., 0] / self.width - 0.5) * 2
        j2d[..., 1] = (j2d[..., 1] / self.height - 0.5) * 2

        return {
            "joints_2d": j2d,     # (T, J, 3)
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


def train_one_epoch(model, loader, optimizer, cfg: Config, scale_factor):
    device = cfg.device
    model.train()

    mpjpe_meter = AverageMeter()

    for batch in loader:    
        gt = batch["joints_3d"].to(device)
        video = batch["joints_2d"].to(device)  # (B, T, J, 3)

        optimizer.zero_grad()   # azzera i gradienti

        pred = model(video)  # (B, T, J, 3)
        
        # --- Root-relative handling ---
        root_gt = gt[:, :, [12], :]
        gt_rel = (gt - root_gt) / scale_factor

        # --- Compute losses ---
        mpjpe = loss_mpjpe(pred, gt_rel)
        loss = mpjpe 

        mpjpe_meter.update(mpjpe.item(), gt.size(0))

        # --- Backward + step ---
        loss.backward()

        optimizer.step()

    results = {"mpjpe": mpjpe_meter.avg}

    return results


def evaluate(model, loader, cfg: Config, scale_factor):
    device = cfg.device
    model.eval()

    mpjpe_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            gt = batch["joints_3d"].to(device)  # (B, T, J, 3)
            video = batch["joints_2d"].to(device)  # (B, T, J, 3)

            # --- Forward ---
            pred = model(video)  # (B, T, J, 3)
    
            # --- Root-relative ---
            root_pred = pred[:, :, [12], :]
            root_gt   = gt[:, :, [12], :]

            pred_rel = (pred - root_pred) * scale_factor
            gt_rel = gt - root_gt

            err1 = loss_mpjpe(pred_rel, gt_rel)

            mpjpe_meter.update(err1.item(), gt.size(0))

    results = {"mpjpe": mpjpe_meter.avg}

    return results


def validate(model, scale_factor, warmup_epochs, patience):
    cfg = Config()
    device = cfg.device

    # Datasets
    train_loader, val_loader = create_dataloaders(
        cfg.datapath,
        batch_size=cfg.batch_size,
        frame_size=cfg.frame_size,
        num_workers=cfg.num_workers,
        val_ratio=0.2,
        seed=cfg.seed
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    #  Scheduler: Warmup + ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', # mode='min': riduce quando la metrica scende (loss o mpjpe)
        factor=0.5, # fdimezza il LR quando scatta (moltiplica per 0.5)
        patience=patience // 2,
        verbose=False,
        min_lr=1e-6
    )

    # 1. Scheduler per il Warmup: sale linearmente per N epoche
    #warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    #    optimizer, 
    #    start_factor=0.1, # Parte dal 10% del LR impostato
    #    end_factor=1.0, 
    #    total_iters=warmup_epochs
    #)
#
    ## 2. Scheduler per il Coseno: parte dopo il warmup
    #cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, 
    #    T_max=cfg.epochs - warmup_epochs, 
    #    eta_min=1e-6
    #)
#
    ## 3. Concatenazione: SequentialLR gestisce lo switch automaticamente
    #scheduler = torch.optim.lr_scheduler.SequentialLR(
    #    optimizer, 
    #    schedulers=[warmup_scheduler, cosine_scheduler], 
    #    milestones=[warmup_epochs]
    #)

    start_epoch = 0
    patience_counter = 0
    best_val = float("inf")

    model = model.to(device)

    # Training
    total_epochs = cfg.epochs

    for epoch in range(start_epoch, total_epochs):
        # --- Gestione Warmup ---
        is_warmup = adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, cfg.lr)

        # --- Training ---
        loss = train_one_epoch(model, train_loader, optimizer, cfg, scale_factor)

        # --- Validation ---
        va = evaluate(model, val_loader, cfg, scale_factor)

        # --- Step dello Scheduler ---
        if not is_warmup:
            scheduler.step(va['mpjpe'])
            #scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            #decay_lr_exponentially(current_lr, optimizer) 

        if va['mpjpe'] < best_val:
            patience_counter = 0
            best_val = va['mpjpe']
        
        patience_counter += 1
        if patience_counter > patience:
            break
            
    return best_val


def objective(trial):
    # -----------------------------------------------------------
    # Hyperparameter da ottimizzare
    # -----------------------------------------------------------
    scale_factor = trial.suggest_float("scale_factor", 1.5, 2.5)
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
    patience = trial.suggest_int("patience", 5, 15)
    depth = trial.suggest_int("depth", 2, 8)

    # -----------------------------------------------------------
    # Istanziazione e Training
    # -----------------------------------------------------------
    try:
        model = optimize_model(depth)

        val_mpjpe = validate(model, scale_factor, warmup_epochs, patience)
        
        return val_mpjpe

    except RuntimeError as e:
        # Se succede qualcosa di strano (es. OOM - Out of Memory)
        print(f"Trial failed with error: {e}")
        # Diciamo a Optuna che questo trial è fallito/pessimo
        return float('inf')


# Avvia lo studio
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Salva risultati su file
with open("optuna_results.txt", "w") as f:
    f.write("Migliori parametri:\n")
    f.write(str(study.best_params) + "\n\n")
    f.write(f"Miglior valore (val_mpjpe): {study.best_value}\n\n")
    f.write("Tutti i trial:\n")
    for i, trial in enumerate(study.trials):
        f.write(f"Trial {i}: {trial.params} -> {trial.value}\n")

print("Migliori parametri:", study.best_params)