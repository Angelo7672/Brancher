"""
Direct 3D Human Pose Estimation from Video (fully supervised)
=============================================================

This script provides a clean, end-to-end PyTorch baseline that predicts 3D skeletons
(J joints × 3 coords) *directly* from RGB video clips — without any 2D keypoint stage.

Highlights
---------
- Input: a window of T frames (B, T, 3, H, W) → Output: (B, T, J, 3)
- Frame encoder: ResNet-18 backbone applied per-frame (batched) to get visual features
- Temporal modeling: TransformerEncoder over sequence of frame features
- Heads: MLP → per-frame 3D joints
- Loss: MSE (MPJPE) + bone-length consistency regularizer (optional)
- Metrics: MPJPE (mm), P-MPJPE (Procrustes aligned, optional)
- Dataset template: `FrameFolderDataset` expecting folders of frames and an `.npz` with 3D labels

Notes
-----
- This is a *reference* baseline. For serious accuracy, consider stronger backbones (VideoMAE, ViT, Swin-V2),
  longer context, multi-view consistency if available, and advanced augmentation.
- Ground truth format expected: joints are **root-centered** (pelvis at 0) and in **millimeters**.
- Cameras/intrinsics are **not** used here (pure 3D pose regression). If your 3D GT is in camera coords,
  keep consistency across train/val.

Author: you + ChatGPT
License: MIT
"""
from __future__ import annotations

import os
import math
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# -------------------------------
# Utilities
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------
# Losses & Metrics
# -------------------------------
def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Per Joint Position Error (mm). pred, gt: (B, T, J, 3)."""
    """Risultato: errore medio di posizione per joint, in millimetri (se i dati sono in mm)."""
    return torch.mean(torch.linalg.norm(pred - gt, dim=-1)) # differenza vettoriale fra posizione predetta e ground truth.


def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor, bones=BONES) -> torch.Tensor: # NEL CASO BONES E' DA RIDEFINIRE
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


# -------------------------------
# Training / Evaluation
# -------------------------------
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


def train_one_epoch(model, loader, optimizer, scaler, cfg: Config, epoch: int):
    model.train()   # set model to training mode
    loss_meter = AverageMeter()
    mpjpe_meter = AverageMeter()

    # Loop sul dataloader
    for video, gt in loader:
        # Itera su batch di clip video + pose ground truth.
        # Sposta tutto su GPU/CPU indicata in cfg.device.
        video = video.to(cfg.device)  # (B, T, 3, H, W)
        gt = gt.to(cfg.device)        # (B, T, J, 3)

        optimizer.zero_grad(set_to_none=True) # resetta i gradienti
        with torch.cuda.amp.autocast(enabled=cfg.amp): # abilita mixed precision se cfg.amp=True.
            pred = model(video) # ottieni le pose 3D predette. (B, T, J, 3)
            loss_pos = F.mse_loss(pred, gt) # MSE loss tra joints predetti e ground truth.
            loss_bone = bone_length_loss(pred, gt) * cfg.lambda_bone # penalizza deviazioni nella lunghezza delle ossa (mantiene proporzioni anatomiche).
            loss = cfg.lambda_mpjpe * loss_pos + loss_bone # combinazione pesata di MPJPE loss e bone loss.

        # Se AMP attivo → usa scaler per scalare gradiente, backprop e step.
        # Se AMP disattivo → fa normale backprop.
        # grad_clip (se non None) → limita la norma dei gradienti per stabilizzare il training.
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

        # Calcola Mean Per Joint Position Error (MPJPE) = distanza media tra joints predetti e ground truth (in mm o unità dataset).
        with torch.no_grad():   # torch.no_grad() → niente gradiente, solo valutazione.
            err = mpjpe(pred, gt)

        # Logging della loss e della metrica. Aggiorna le medie di loss e MPJPE, pesate per batch size.
        bs = video.size(0)
        loss_meter.update(loss.item(), bs)
        mpjpe_meter.update(err.item(), bs)

    return {"loss": loss_meter.avg, "mpjpe": mpjpe_meter.avg}


def evaluate(model, loader, cfg: Config):
    model.eval()    # set model to evaluation mode
    mpjpe_meter = AverageMeter()  # tiene traccia della media di MPJPE sui batch.
    with torch.no_grad():   # torch.no_grad() → nessun calcolo dei gradienti, quindi più veloce e meno memoria.
        for video, gt in loader:
            video = video.to(cfg.device)    # clip di input (B,T,3,H,W)
            gt = gt.to(cfg.device)  # 3D ground truth (B,T,J,3)
            pred = model(video) # predizioni (B,T,J,3)
            err = mpjpe(pred, gt)   # errore medio per joint su quel batch
            mpjpe_meter.update(err.item(), video.size(0))   # aggiorna la media pesata per batch size.
    return {"mpjpe": mpjpe_meter.avg}


# -------------------------------
# Main
# -------------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)

    # Datasets
    train_ds = 

    val_ds   = 

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model = Direct3DPoseNet(
        num_joints=cfg.num_joints,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
    ).to(cfg.device)

    print(f"Model params: {count_params(model)/1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # AMP scaler, Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Simple warmup. Funzione che calcola un fattore moltiplicativo per il learning rate ad ogni step.
    # Idea: partire con LR piccolo e aumentarlo gradualmente per i primi warmup_epochs.
    def lr_lambda(it):
        if cfg.warmup_epochs > 0:
            total_steps = len(train_loader) * cfg.warmup_epochs
            return min(1.0, (it+1) / max(1, total_steps))
        return 1.0 # Se warmup_epochs=0, ritorna sempre 1.0 (nessun warmup).
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) # scheduler che applica lr_lambda ad ogni step.

    # In pratica:
    #    AdamW aggiorna i pesi.
    #    AMP accelera il training con mixed precision.
    #    Warmup evita di “spaventare” la rete con un LR troppo alto all’inizio.

    best_val = float('inf')

    for epoch in range(cfg.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg, epoch)
        va = evaluate(model, val_loader, cfg)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train loss {tr['loss']:.4f} mpjpe {tr['mpjpe']:.2f}mm | val mpjpe {va['mpjpe']:.2f}mm")

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
            }, f"checkpoints/best.pth")
            print(f"Saved best checkpoint (val MPJPE={best_val:.2f}mm)")


if __name__ == "__main__":
    # Usage:
    # 1) Prepare frame folders and matching .npz files with 'joints3d'.
    # 2) Write JSON index files (train/val) with entries as documented.
    # 3) Run:  python direct_3d_hpe.py
    main()
