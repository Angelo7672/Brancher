import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from glob import glob
import pickle
from torchinfo import summary
from datetime import datetime
import time
import argparse

from model import create_model
from config import Config
from loss import *
from rotations import *
from learning import *


# -----------------------------------------------------------------------------
# Dataset & DataLoader
# -----------------------------------------------------------------------------
class AthletePosePreprocessed(Dataset):
    def __init__(self, data_dir):
        self.samples = sorted(glob(os.path.join(data_dir, "*.pkl")))
        if len(self.samples) == 0:
            print(f"Warning: No .pkl files found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)
        
        j2d = torch.tensor(data['data_input'], dtype=torch.float32)
        j3d = torch.tensor(data['data_label'], dtype=torch.float32)
        
        return {
            "joints_2d": j2d,
            "joints_3d": j3d,
        }


def create_full_train_loader(cfg: Config, test_path: str):
    print("Preparing the final dataset (Train + Valid + Test)...")
    
    # Paths from config and user input
    path_train = os.path.join(cfg.datapath_athlete, "train")
    path_valid = os.path.join(cfg.datapath_athlete, "valid")
    path_test  = os.path.join(test_path, "test")
    
    # Initialize individual datasets
    ds_train = AthletePosePreprocessed(path_train)
    ds_valid = AthletePosePreprocessed(path_valid)
    ds_test  = AthletePosePreprocessed(path_test)
    
    # Concatenate all available data
    full_ds = ConcatDataset([ds_train, ds_valid, ds_test])
    
    loader = DataLoader(
        full_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Total samples for final training: {len(full_ds)}")
    return loader


# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------
def decay_lr_exponentially(lr, optimizer, lr_decay=0.99):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr

def train_one_epoch(model, loader, optimizer, loss_wrapper, epoch, cfg: Config):
    device = cfg.device
    model.train()

    # Initialize loss meters
    mpjpe_meter = AverageMeter()
    bone_meter = AverageMeter()
    velocity_meter = AverageMeter()
    acceleration_meter = AverageMeter()
    geodesic_meter = AverageMeter()
    vector_meter = AverageMeter()
    hm_meter = AverageMeter()
    
    batch_idx = 0
    for batch in loader:
        print(f" Batch {batch_idx+1}/{len(loader)}", end='\r')
        batch_idx += 1

        gt = batch["joints_3d"].to(device)
        video = batch["joints_2d"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # pos: (B, T, J, 3), rot: (B, T, 16, 3, 3), htmp: (B, T, J, H, W), log_var_hm: (B, T, J)
        pos, rot, htmp, log_var_hm = model(video)
        
        # Root-relative handling
        root_gt = gt[:, :, 0:1, :]
        gt_rel = gt - root_gt

        # Rotation Branch Helper
        t_pose = get_t_pose_h36m()
        bones = get_h36m_bones(pos)
        r_p = get_rotation_between_vectors(t_pose, bones)

        # Compute multi-task losses
        losses = {
            'mpjpe': loss_mpjpe(pos, gt_rel),
            'bone': loss_limb_var(pos),
            'velocity': velocity_smoothness_loss(pos),
            'acceleration': acceleration_smoothness_loss(pos),
            'geodesic': geodesic_loss(rot, r_p),
            'vector': bone_vector_loss(rot, pos),
            'hm': aleatoric_heatmap_loss(htmp, log_var_hm, video)
        }

        # Loss balancing and backpropagation
        total_loss, current_weights = loss_wrapper(losses, epoch)
        total_loss.backward()
        optimizer.step()

        # Stats update
        mpjpe_meter.update(losses['mpjpe'].item(), video.size(0))
        bone_meter.update(losses['bone'].item(), video.size(0))
        velocity_meter.update(losses['velocity'].item(), video.size(0))
        acceleration_meter.update(losses['acceleration'].item(), video.size(0))
        geodesic_meter.update(losses['geodesic'].item(), video.size(0))
        vector_meter.update(losses['vector'].item(), video.size(0))
        hm_meter.update(losses['hm'].item(), video.size(0))

    results = {
        "mpjpe": mpjpe_meter.avg, "bone": bone_meter.avg, "velocity": velocity_meter.avg, 
        "acceleration": acceleration_meter.avg, "geodesic": geodesic_meter.avg, 
        "vector": vector_meter.avg, "heatmap": hm_meter.avg, "weights": current_weights
    }
    return results


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Final training on full AthletePose dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a specific checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to the .pth checkpoint")
    args = parser.parse_args()
    
    cfg = Config()
    # Path for the test set frames as requested
    test_set_path = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/test/frame_81"

    # Unified DataLoader
    train_loader = create_full_train_loader(cfg, test_set_path)

    # Set up experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(cfg.base_dir, f"FINAL_model_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "final_training_log.txt")

    # Initialize Model
    model = create_model()
    
    start_epoch = 0
    if args.resume and args.resume_path:
        print(f"Resuming training from: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("Loading pretrained weights from MotionAGFormer...")
        model = load_pretrained_weights(model, cfg.pretrained_weights, cfg.device)

    # Loss Wrapper & Optimizer initialization
    loss_wrapper = MultiTaskLossWrapperPretrain(total_epochs=cfg.epochs).to(cfg.device)
    if args.resume and 'loss_wrapper' in checkpoint:
        loss_wrapper.load_state_dict(checkpoint['loss_wrapper'])

    # Prepare trainable parameters (Model + Uncertainty weights in Loss Wrapper)
    trainable_params = list(model.parameters()) + list(loss_wrapper.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Utilizing {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(cfg.device)

    # Initial Log Entry
    with open(log_path, "w") as f:
        f.write(f"==== DEFINITIVE TRAINING STARTED AT {timestamp} ====\n")
        f.write(f"Total dataset size: {len(train_loader.dataset)}\n")
        f.write(f"Config: Epochs={cfg.epochs}, Base LR={cfg.lr}\n\n")

    # Final Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        
        # Execute Epoch
        metrics = train_one_epoch(model, train_loader, optimizer, loss_wrapper, epoch, cfg)
        
        # LR Decay
        decay_lr_exponentially(optimizer.param_groups[0]['lr'], optimizer)

        # Save Checkpoint (Keeps current and previous to save storage)
        ckpt_path = os.path.join(exp_dir, f"final_weight_epoch{epoch+1}.pth")
        state = {
            'model': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
            'loss_wrapper': loss_wrapper.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'cfg': cfg.__dict__
        }
        torch.save(state, ckpt_path)
        
        # Delete checkpoint from two epochs ago to maintain clean storage
        old_ckpt = os.path.join(exp_dir, f"final_weight_epoch{epoch-1}.pth")
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

        # Logging results
        epoch_time_min = (time.time() - epoch_start) / 60
        current_lr = optimizer.param_groups[0]['lr']
        log_line = (f"Epoch {epoch+1:03d}/{cfg.epochs} | LR: {current_lr:.8f} | "
                    f"MPJPE: {metrics['mpjpe']:.6f} | Bone: {metrics['bone']:.6f} | "
                    f"Heatmap Loss: {metrics['heatmap']:.6f} | Time: {epoch_time_min:.2f}min")
        
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

    print(f"Definitive training complete. Final weights saved in: {exp_dir}")


if __name__ == "__main__":
    main()