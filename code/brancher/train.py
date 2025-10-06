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


def train_one_epoch(model, loader, optimizer, losses, cfg: Config):
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
        #gt = gt.to(cfg.device, dtype=torch.float32)
        batch_gt = gt.cuda()

        #with torch.cuda.amp.autocast(enabled=cfg.amp):
        with torch.no_grad():
            # Itera sulle viste senza concatenare
            for view in views:
                batch_input = batch["video"]["image"][view]  # (B, T, H, W, 3)
                if isinstance(batch_input, np.ndarray):
                    batch_input = torch.from_numpy(batch_input).float()
                #video = video.to(cfg.device, dtype=torch.float32)
                batch_input = batch_input.cuda()

                # DA METTERE A POSTO QUESTO ARG
                if args.rootrel:
                    batch_gt = batch_gt - batch_gt[:,:,0:1,:]
                else:
                    batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0.
                if args.mask or args.noise:
                    batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)

                predicted_3d_pos = model(batch_inpu)    # (B, T, J, 3) 

                optimizer.zero_grad()

                #Loss
                loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
                loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
                loss_lv = loss_limb_var(predicted_3d_pos)
                loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
                loss_a = loss_angle(predicted_3d_pos, batch_gt)
                loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
                loss_total = loss_3d_pos + \
                             args.lambda_scale       * loss_3d_scale + \
                             args.lambda_3d_velocity * loss_3d_velocity + \
                             args.lambda_lv          * loss_lv + \
                             args.lambda_lg          * loss_lg + \
                             args.lambda_a           * loss_a  + \
                             args.lambda_av          * loss_av
                losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
                losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
                losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
                losses['lv'].update(loss_lv.item(), batch_size)
                losses['lg'].update(loss_lg.item(), batch_size)
                losses['angle'].update(loss_a.item(), batch_size)
                losses['angle_velocity'].update(loss_av.item(), batch_size)
                losses['total'].update(loss_total.item(), batch_size)
                
                loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
                loss_total = loss_2d_proj
                losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
                losses['total'].update(loss_total.item(), batch_size)

                loss_total.backward()
                optimizer.step()


def evaluate(model, loader, cfg: Config):
    model.eval()
    mpjpe_meter = AverageMeter()
    p_mpjpe_meter = AverageMeter()

    print("I'm evaluating")
    i = 0

    # Lista delle viste disponibili
    views = ['front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left']

    with torch.no_grad():
        for batch in loader:
            print(f" Batch {i+1}/{len(loader)}", end='\r')
            i += 1
            # Ground truth 3D pose (B, T, J, 3)
            gt = batch["joints_3d"]["data_points"]
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt).float()
            #gt = gt.to(cfg.device, dtype=torch.float32)
            batch_gt = gt.cuda()

            # Itera sulle viste senza concatenare
            for view in views:
                video = batch["video"]["image"][view]  # (B, T, H, W, 3)
                if isinstance(video, np.ndarray):
                    video = torch.from_numpy(video).float()
                #video = video.to(cfg.device, dtype=torch.float32)
                batch_input = batch_input.cuda()

                predicted_3d_pos = model(video)

                if args.rootrel:
                    predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
                else:
                    batch_gt[:,0,0,2] = 0

                if args.gt_2d:
                    predicted_3d_pos[...,:2] = batch_input[...,:2]

                # Errore Protocol #1
                err1 = mpjpe(pred, gt)

                # Errore Protocol #2 (Procrustes alignment)
                err2 = p_mpjpe(pred, gt)




                

                # Aggiorna il meter usando la dimensione reale del batch (non ×7)
                batch_gt = gt.size(0)
                mpjpe_meter.update(err1.item(), bs)
                p_mpjpe_meter.update(err2.item(), bs)
    
    results = {
        "mpjpe": mpjpe_meter.avg,
        "p_mpjpe": p_mpjpe_meter.avg,
    }

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

    model = create_model()

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    #CARICA PESI MOTIONBERT

    # AMP scaler, Mixed Precision Training
    #scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

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
        losses = {}
        losses['3d_pos'] = AverageMeter()
        losses['3d_scale'] = AverageMeter()
        losses['2d_proj'] = AverageMeter()
        losses['lg'] = AverageMeter()
        losses['lv'] = AverageMeter()
        losses['total'] = AverageMeter()
        losses['3d_velocity'] = AverageMeter()
        losses['angle'] = AverageMeter()
        losses['angle_velocity'] = AverageMeter()
        
        train_one_epoch(model, train_loader, optimizer, losses, cfg)

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

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay


        log_line = f"Epoch {epoch+1:03d}/{total_epochs} | train loss {losses['total']:.4f} | val mpjpe {va['mpjpe']:.2f} | val p_mpjpe {va['p_mpjpe']:.2f}"
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