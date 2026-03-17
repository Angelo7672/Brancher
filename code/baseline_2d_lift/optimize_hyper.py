import optuna
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pickle

from config import Config
from loss import *
from learning import AverageMeter
from model import optimize_model, create_model


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
    """Only handles the first N epochs of linear warmup"""
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


def train_one_epoch(model, loader, optimizer, cfg: Config):
    device = cfg.device
    model.train()

    mpjpe_meter = AverageMeter()

    for batch in loader:    
        gt = batch["joints_3d"].to(device)
        video = batch["joints_2d"].to(device)  # (B, T, J, 3)

        optimizer.zero_grad()

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


def evaluate(model, loader, cfg: Config):
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
            root_pred = pred[:, :, [0], :]
            root_gt   = gt[:, :, [0], :]

            pred_rel = pred - root_pred
            gt_rel = gt - root_gt

            err1 = loss_mpjpe(pred_rel, gt_rel)

            mpjpe_meter.update(err1.item(), gt.size(0))

    results = {"mpjpe": mpjpe_meter.avg}

    return results


def validate(model, warmup_epochs, patience):
    cfg = Config()
    device = cfg.device

    # Datasets
    train_loader, val_loader = create_dataloaders(
        cfg.datapath_athlete,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    patience_counter = 0
    best_val = float("inf")

    model = model.to(device)

    # Training
    total_epochs = cfg.epochs
    counter = 0

    for epoch in range(start_epoch, total_epochs):
        counter += 1
        # --- Gestione Warmup ---
        is_warmup = adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, cfg.lr)

        # --- Training ---
        loss = train_one_epoch(model, train_loader, optimizer, cfg)

        # --- Validation ---
        va = evaluate(model, val_loader, cfg)

        # --- Step dello Scheduler ---
        if not is_warmup:
            current_lr = optimizer.param_groups[0]['lr']
            decay_lr_exponentially(current_lr, optimizer)

        if va['mpjpe'] < best_val:
            patience_counter = 0
            best_val = va['mpjpe']
        
        patience_counter += 1
        if patience_counter > patience:
            break
            
    return best_val, counter


def objective(trial):
    # Hyperparameters
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
    patience = trial.suggest_int("patience", 3, 10)
    depth = trial.suggest_int("depth", 2, 8)

    try:
        model = create_model()

        val_mpjpe, counter = validate(model, warmup_epochs, patience)

        trial.set_user_attr("counter", counter)
        
        return val_mpjpe

    except RuntimeError as e:
        print(f"Trial failed with error: {e}")
        return float('inf')


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

with open("optuna_results.txt", "w") as f:
    f.write("Best parameters:\n")
    f.write(str(study.best_params) + "\n\n")
    f.write(f"Best value (val_mpjpe): {study.best_value}\n\n")
    f.write("All trials:\n")
    for i, trial in enumerate(study.trials):
        counter_val = trial.user_attrs.get('counter', 'N/A')
        f.write(f"Trial {i}: {trial.params} -> {trial.value}, epochs: {counter_val}\n")

print("Best parameters:", study.best_params)