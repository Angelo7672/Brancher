import torch
import json
import os
import shutil

# Mappatura corretta per lo swap destra/sinistra (basata su standard COCO/SportsPose)
LEFT_RIGHT_PAIRS = [
    (1, 2),   # Occhi
    (3, 4),   # Orecchie
    (5, 6),   # Spalle
    (7, 8),   # Gomiti
    (9, 10),  # Polsi
    (11, 12), # Fianchi
    (13, 14), # Ginocchia
    (15, 16)  # Caviglie
]
# Il punto 0 (naso) è centrale e non serve scambiarlo.


def flip_data(image_w, heatmaps_path, kpts2d_path, kpts3d_path, output_dir):
    # --- 1. HEATMAPS (.pt) ---
    hm = torch.load(heatmaps_path)  # Shape: [T, J, H, W]
    # Flip orizzontale lungo l'asse W (dimensione 3)
    hm_flipped = torch.flip(hm, dims=[3])

    # Swap dei canali J (dimensione 1) per tutti i frame T simultaneamente
    for l, r in LEFT_RIGHT_PAIRS:
        # Usiamo lo slicing di PyTorch per uno swap efficiente in-place
        temp = hm_flipped[:, l].clone()
        hm_flipped[:, l] = hm_flipped[:, r]
        hm_flipped[:, r] = temp
        
    torch.save(hm_flipped, os.path.join(output_dir, 'heatmaps.pt'))

    # --- 2. KEYPOINTS 2D (.json) ---
    with open(kpts2d_path, 'r') as f:
        k2d_seq = json.load(f) # Lista di [T, J, 3]] x,y,conf
    
    new_k2d_seq = []
    for frame in k2d_seq:
        # Flip geometrico: x_new = W - x_old
        # Manteniamo y (p[1]) e confidence (p[2]) invariati
        flipped_frame = [[image_w - p[0], p[1], p[2]] for p in frame]
        
        # Swap semantico dei giunti
        for l, r in LEFT_RIGHT_PAIRS:
            flipped_frame[l], flipped_frame[r] = flipped_frame[r], flipped_frame[l]
        new_k2d_seq.append(flipped_frame)
        
    with open(os.path.join(output_dir, 'keypoints2d.json'), 'w') as f:
        json.dump(new_k2d_seq, f)

    # --- 3. KEYPOINTS 3D (.json) ---
    with open(kpts3d_path, 'r') as f:
        k3d_seq = json.load(f) # Lista di [T, J, 3] x,y,z
    
    new_k3d_seq = []
    for frame in k3d_seq:
        # Flip geometrico 3D: invertiamo l'asse X
        flipped_frame = [[-p[0], p[1], p[2]] for p in frame]
        
        # Swap semantico dei giunti
        for l, r in LEFT_RIGHT_PAIRS:
            flipped_frame[l], flipped_frame[r] = flipped_frame[r], flipped_frame[l]
        new_k3d_seq.append(flipped_frame)

    with open(os.path.join(output_dir, 'keypoints3d.json'), 'w') as f:
        json.dump(new_k3d_seq, f)



src = "/Experiments/angeloa00/AirPose_COCO/train"
dst = "/Experiments/angeloa00/AirPose_COCO/train_flipped"
frame_size = (640, 352)  # Per AirPose
block_id = 148

os.makedirs(dst, exist_ok=True)

folders = sorted([d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))])

for folder in folders:
    src_folder = os.path.join(src, folder)
    dst_folder = os.path.join(dst, folder)  # Create destination folder for the current folder
    shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True) # Copy the entire folder and its contents
    dst_folder = os.path.join(dst, f"block_{block_id}")
    os.makedirs(dst_folder, exist_ok=True)

    heatmaps_path = os.path.join(src_folder, 'heatmaps.pt')
    kpts2d_path = os.path.join(src_folder, 'keypoints2d.json')
    kpts3d_path = os.path.join(src_folder, 'keypoints3d.json')

    flip_data(frame_size[1], heatmaps_path, kpts2d_path, kpts3d_path, dst_folder)
    print(f"Processed and saved flipped data for: {folder}")
    
    block_id += 1