import os
import torch
import torch.nn as nn
import argparse
import math
import json
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pickle
from tqdm import tqdm
import numpy as np
import copy

from model import create_model
from datareader_ap3d import DataReaderH36M
from config import Config
from loss import *
from learning import load_pretrained_weights, AverageMeter

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
        
        # Ritorna sempre un dizionario con le chiavi standard
        sample = {
            "joints_2d": j2d,
            "joints_3d": j3d,
        }

        return sample

def create_test_dataloader(data_dir, batch_size, num_workers):
    print(f"Loading AthletePose Test Set from {data_dir}")
    test_ds = AthletePosePreprocessed(data_dir)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data

def evaluate(model, loader, datareader, cfg):
    print("I'm testing")
    results_all = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["joints_2d"].to(cfg.device)
            y = batch["joints_3d"].to(cfg.device)

            batch_input_flip = flip_data(x)
            predicted_3d_pos_1 = model(x)
            predicted_3d_pos_flip = model(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2

            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    jpe_all = np.zeros((num_test_frames, cfg.num_joints))
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(cfg.num_joints)]
    results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
        for joint_idx in range(cfg.num_joints):
            results_joints[joint_idx][action] = []

    for idx in range(len(action_clips)):
        source = source_clips[idx][:-6]
        frame_list = frame_clips[idx]
        action = action_clips[idx]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = mpjpe(pred, gt)
        jpe = per_jpe(pred, gt)
        for joint_idx in range(cfg.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        e1_all[frame_list] += err1
        err2 = p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results_procrustes[action].append(err2)
            results[action].append(err1)
            for joint_idx in range(cfg.num_joints):
                jpe = jpe_all[idx, joint_idx] / oc[idx]
                results_joints[joint_idx][action].append(jpe)
    final_result_procrustes = []
    final_result_joints = [[] for _ in range(cfg.num_joints)]
    final_result = []

    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        for joint_idx in range(cfg.num_joints):
            final_result_joints[joint_idx].append(np.mean(results_joints[joint_idx][action]))

    joint_errors = []
    for joint_idx in range(cfg.num_joints):
        joint_errors.append(
            np.mean(np.array(final_result_joints[joint_idx]))
        )
    joint_errors = np.array(joint_errors)
    e1 = np.mean(np.array(final_result))
    assert round(e1, 4) == round(np.mean(joint_errors), 4), f"MPJPE {e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    e2 = np.mean(np.array(final_result_procrustes))
    
    results = {"mpjpe": e1, "p_mpjpe": e2}
    return results

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    checkpoint_path = "experiments/motionagformer/motionagformer-s-ap3d.pth.tr"
    test_dir = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/pose_3d_v3/frame_81/valid"

    datareader = DataReaderH36M(n_frames=cfg.clip_len, sample_stride=1, data_stride_train=cfg.clip_len // 3, data_stride_test=cfg.clip_len,
                                dt_root="/mnt/proj2/eu-25-52/ar17/AthletePose3D/pose_3d_v3")

    # 1. Caricamento Modello
    model = create_model(cfg.frame_size_sport)
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    model = load_pretrained_weights(model, checkpoint_path, device)
    
    model.to(device)

    # 2. Setup Data
    test_loader = create_test_dataloader(
        test_dir,
        cfg.batch_size, 
        cfg.num_workers
    )

    # 3. Esecuzione Test
    results = evaluate(model, test_loader, datareader, cfg)

    # 4. Logging Risultati
    output_str = (
        f"\n{'='*30}\n"
        f"TEST RESULTS\n"
        f"{'='*30}\n"
        f"Checkpoint: {os.path.basename(checkpoint_path)}\n"
        f"Total Samples: {len(test_loader.dataset)}\n"
        f"MPJPE:   {results['mpjpe']:.4f} mm\n"
        f"P-MPJPE: {results['p_mpjpe']:.4f} mm\n"
        f"{'='*30}\n"
    )
    
    print(output_str)

    # Salva i risultati in un file nella stessa cartella del checkpoint
    res_path = os.path.join(os.path.dirname(checkpoint_path), "test_results.txt")
    with open(res_path, "w") as f:
        f.write(output_str)
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()