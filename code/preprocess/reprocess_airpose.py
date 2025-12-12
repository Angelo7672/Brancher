import os
import cv2
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import shutil 
from tqdm import tqdm

# === Percorsi base ===
root = "/DATA/AirPose/copenet_dji_real_data"
machine_dirs = [os.path.join(root, "machine_1"), os.path.join(root, "machine_2")]

# === Funzioni helper ===
def read_calibration(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    D = cv_file.getNode("D").mat()
    cv_file.release()
    return K, D


def fuse_keypoints(openpose_data, alphapose_data, frame_id):
    try:
        # === Estraggo OpenPose ===
        pose_op = None
        if frame_id in openpose_data and isinstance(openpose_data[frame_id], dict):
            pose_field = openpose_data[frame_id].get('pose', [])
            if isinstance(pose_field, list) and len(pose_field) > 0:
                arr = np.array(pose_field[0])
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    pose_op = arr[:, :2]
            else:
                print(f"[WARN] Frame {frame_id}: OpenPose vuoto o non valido.")

        # === Estraggo AlphaPose ===
        pose_ap = None
        if frame_id in alphapose_data and isinstance(alphapose_data[frame_id], dict):
            ap_people = alphapose_data[frame_id].get('people', [])
            if len(ap_people) > 0:
                raw_ap = np.array(
                    ap_people[0].get('pose_keypoints_2d', ap_people[0].get('keypoints', []))
                )
                if raw_ap.size > 0:
                    pose_ap = raw_ap.reshape(-1, 3)[:, :2]
            else:
                print(f"[WARN] Frame {frame_id}: AlphaPose nessuna persona.")
        else:
            print(f"[WARN] Frame {frame_id}: chiave mancante in AlphaPose.")

        # === Gestione mancanze ===
        if pose_op is None and pose_ap is None:
            print(f"[WARN] Frame {frame_id}: nessun keypoint disponibile (OpenPose + AlphaPose mancanti).")
            return np.zeros((17, 2), dtype=float)
        elif pose_op is None:
            print(f"[INFO] Frame {frame_id}: uso solo AlphaPose.")
            # Mapping AlphaPose18 → COCO17
            ap_to_coco = np.array([0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10], dtype=int)
            return pose_ap[ap_to_coco]
        elif pose_ap is None:
            print(f"[INFO] Frame {frame_id}: uso solo OpenPose.")
            # Mapping OpenPose25 → COCO17
            op_to_coco = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11], dtype=int)
            return pose_op[op_to_coco]

        # === Se entrambe presenti, fondi ===
        op_to_coco = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11], dtype=int)
        ap_to_coco = np.array([0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10], dtype=int)

        pose_op_common = pose_op[op_to_coco]
        pose_ap_common = pose_ap[ap_to_coco]

        # Media robusta
        pose_avg = np.zeros_like(pose_op_common)
        for i in range(17):
            op, ap = pose_op_common[i], pose_ap_common[i]
            if np.allclose(op, 0) and np.allclose(ap, 0):
                pose_avg[i] = [0, 0]
            elif np.allclose(op, 0):
                pose_avg[i] = ap
            elif np.allclose(ap, 0):
                pose_avg[i] = op
            else:
                pose_avg[i] = (op + ap) / 2.0

        return pose_avg

    except Exception as e:
        print(f"[ERROR] Frame {frame_id}: errore inatteso in fuse_keypoints() → {e}")
        return np.zeros((17, 2), dtype=float)


def resize_joints_2d(joints, orig_h, orig_w, target_size):
    target_h, target_w = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    joints_resized = joints.clone()
    joints_resized[..., 0] *= scale_x
    joints_resized[..., 1] *= scale_y
    return joints_resized


def create_video_from_frames(img_dir, start_frame, end_frame):
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    frame_files = frame_files[start_frame:end_frame+1]
    frames = []
    for f in frame_files:
        img_bgr = cv2.imread(os.path.join(img_dir, f))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
    return np.stack(frames, axis=0)


def make_frame_ids(global_start, global_end):
    frame_ids = [f"{i:06d}" for i in range(global_start, global_end + 1)]
    return frame_ids


def temporal_smooth_and_fill(poses_2d_seq, window=5):
    smoothed = poses_2d_seq.copy().astype(float)
    T = smoothed.shape[0]
    for k in range(17):
        for dim in range(2):
            coords = smoothed[:, k, dim]
            mask_valid = coords != 0
            if np.any(mask_valid):
                coords[~mask_valid] = np.interp(
                    np.flatnonzero(~mask_valid),
                    np.flatnonzero(mask_valid),
                    coords[mask_valid]
                )
            else:
                coords[:] = 0
            kernel = np.ones(window) / window
            coords_smooth = np.convolve(coords, kernel, mode='same')
            smoothed[:, k, dim] = coords_smooth
    return smoothed


# === Parametri ===
BLOCK_SIZE = 243
DEFAULT_WINDOW = 5
TARGET_SIZE = (640, 352) # W, H

# --- Frame range personalizzabile ---  # Train set: 7580-25561
START_FRAME = 7580       # 👈 imposta qui il frame di partenza
END_FRAME   = 25561    # 👈 imposta qui il frame finale (inclusivo)
assert (END_FRAME - START_FRAME + 1) % BLOCK_SIZE == 0, \
       "⚠️ L'intervallo deve essere multiplo di 243!"

# =========================================================================
# === PRE-PROCESSING SETUP: Creazione Cartelle e Scaling Parametri Camera ===
# =========================================================================
experiment_root = "/Experiments/angeloa00/AirPose"

print("\n=== ⚙️ Setup Environment & Scaling Calibration ===")

for i, machine_dir in enumerate(machine_dirs):
    machine_name = f"machine_{i+1}"
    dest_dir = os.path.join(experiment_root, machine_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Carica Calibrazione Originale
    calib_path = os.path.join(machine_dir, "camera_calib.yml")
    if os.path.exists(calib_path):
        K_orig, D_orig = read_calibration(calib_path)
        
        # 2. Leggi un'immagine per ottenere H e W originali
        img_folder = os.path.join(machine_dir, "images")
        sample_img_name = sorted(os.listdir(img_folder))[0]
        sample_img = cv2.imread(os.path.join(img_folder, sample_img_name))
        orig_h, orig_w = sample_img.shape[:2]
        
        # 3. Calcola fattori di scala
        scale_x = TARGET_SIZE[0] / orig_w
        scale_y = TARGET_SIZE[1] / orig_h
        
        # 4. Scala K (f_x, f_y, c_x, c_y)
        K_new = K_orig.copy()
        K_new[0, 0] *= scale_x # fx
        K_new[0, 2] *= scale_x # cx
        K_new[1, 1] *= scale_y # fy
        K_new[1, 2] *= scale_y # cy
        
        # 5. Salva Nuova Calibrazione nella destinazione
        save_calib_path = os.path.join(dest_dir, "camera_calib.yml")
        cv_file = cv2.FileStorage(save_calib_path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("K", K_new)
        cv_file.write("D", D_orig) # D non cambia col resize semplice (solitamente)
        cv_file.release()
        print(f"[INFO] {machine_name}: Saved SCALED calibration to {save_calib_path}")
        
    # 6. Copia markerposes (Le pose estrinseche sono unità fisiche, non scalano col pixel)
    marker_path = os.path.join(machine_dir, "markerposes_corrected_all.pkl")
    if os.path.exists(marker_path):
        dest_marker_path = os.path.join(dest_dir, "markerposes_corrected_all.pkl")
        shutil.copy(marker_path, dest_marker_path)
        print(f"[INFO] {machine_name}: Copied marker poses to {dest_marker_path}")

print("=== Setup Complete ===\n")

# =========================================================================
# === Loop principale ===
# =========================================================================
block_counter = 0

for global_start in range(START_FRAME, END_FRAME + 1, BLOCK_SIZE):
    global_end = global_start + BLOCK_SIZE - 1
    print(f"\n=== 🔹 Processing block {block_counter} ({global_start}–{global_end}) ===")

    poses_2d_blocks = []
    video_blocks = []
    Ks_list, Rs_list, ts_list = [], [], []

    for machine_dir in machine_dirs:
        #with open(os.path.join(machine_dir,"openpose_res.pkl"),"rb") as f:
        #    openpose_data = pickle.load(f)
        #with open(os.path.join(machine_dir,"alphapose_res.json"),"r") as f:
        #    alphapose_data = json.load(f)
        #with open(os.path.join(machine_dir,"markerposes_corrected_all.pkl"),"rb") as f:
        #    markers = pickle.load(f)
        #K, _ = read_calibration(os.path.join(machine_dir,"camera_calib.yml"))
#
        #block_frame_ids = make_frame_ids(global_start, global_end)


        video = create_video_from_frames(os.path.join(machine_dir,"images"), global_start, global_end)
        video_blocks.append(video)

        #poses_2d = []
        #Ks, Rs, ts = [], [], []
#
        #for frame_id in block_frame_ids:
        #    fused = fuse_keypoints(openpose_data, alphapose_data, frame_id)
        #    poses_2d.append(fused)
#
        #    Ks.append(K)
        #    rvec = np.array(markers[frame_id]['0']['rvec'])
        #    tvec = np.array(markers[frame_id]['0']['tvec']).reshape(3,)
        #    R, _ = cv2.Rodrigues(rvec)
        #    Rs.append(R)
        #    ts.append(tvec.reshape(3,1))
#
        #poses_2d = np.stack(poses_2d, axis=0)  # (T,17,2)

        # === Padding dei frame vuoti usando l'ultimo frame valido ===
        #last_valid = None
        #for t in range(poses_2d.shape[0]):
        #    if np.allclose(poses_2d[t], 0.0, atol=1e-8):
        #        if last_valid is not None:
        #            poses_2d[t] = last_valid.copy()
        #        # se last_valid è None, lasciamo zero
        #    else:
        #        last_valid = poses_2d[t].copy()

        # === Smoothing ===
        #poses_2d = temporal_smooth_and_fill(poses_2d, window=DEFAULT_WINDOW)
#
        #poses_2d_blocks.append(poses_2d)
        #Ks_list.append(Ks)
        #Rs_list.append(Rs)
        #ts_list.append(ts)

    # --- Triangolazione 3D ---
    #poses_3d = []
    #num_frames = min(poses_2d_blocks[0].shape[0], poses_2d_blocks[1].shape[0])
    #print(f"🟢 Processing {num_frames} frames (min length among both views)")
#
    #for fidx in range(num_frames):
    #    mask_common = ~(np.all(np.isclose(poses_2d_blocks[0][fidx],0.0,atol=1e-8),axis=1)) & \
    #                  ~(np.all(np.isclose(poses_2d_blocks[1][fidx],0.0,atol=1e-8),axis=1))
    #    valid_indices = np.where(mask_common)[0]
#
    #    pose1_all = poses_2d_blocks[0][fidx][valid_indices].T
    #    pose2_all = poses_2d_blocks[1][fidx][valid_indices].T
#
    #    P1 = Ks_list[0][fidx] @ np.hstack((Rs_list[0][fidx], ts_list[0][fidx]))
    #    P2 = Ks_list[1][fidx] @ np.hstack((Rs_list[1][fidx], ts_list[1][fidx]))
#
    #    if valid_indices.size > 0:
    #        pts4d_h = cv2.triangulatePoints(P1, P2, pose1_all, pose2_all)
    #        pts3d = (pts4d_h[:3] / pts4d_h[3]).T
    #        pts3d_full = np.zeros((17,3), dtype=float)
    #        pts3d_full[valid_indices] = pts3d
    #    else:
    #        pts3d_full = np.zeros((17,3), dtype=float)
    #    poses_3d.append(pts3d_full)
#
    #poses_3d = torch.from_numpy(np.stack(poses_3d, axis=0)).float()  # (T,17,3)

    output_root = "/Experiments/angeloa00/AirPose/train"
    os.makedirs(output_root, exist_ok=True)

    #for video_np, pose2d_np in zip(video_blocks, poses_2d_blocks):
    for video_np in video_blocks:
        # === Resize video ===
        v = torch.from_numpy(video_np).permute(0,3,1,2).float()  # (T,C,H,W)
        v = F.interpolate(v, size=(352, 640), mode="bilinear", align_corners=False)  # H,W
        v = v.permute(0,2,3,1).byte().numpy()  # torna numpy (T,H,W,C)

        # === Resize joints ===
        #T, orig_h, orig_w, _ = video_np.shape
        #j2d = torch.from_numpy(pose2d_np).float()
        #j2d = resize_joints_2d(j2d, orig_h, orig_w, (352, 640))
        #j2d = j2d.numpy().tolist()

        # === Convert 3d joints to list ===
        #j3d = poses_3d.numpy().tolist()

        # === Create output folder ===
        block_name = f"block_{block_counter:03d}"
        block_dir = os.path.join(output_root, block_name)
        os.makedirs(block_dir, exist_ok=True)

        # === Save video.mp4 ===
        video_path = os.path.join(block_dir, "video.mp4")
        print(f"[INFO] Saving video → {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 24, (640, 352))

        for frame in v:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

        # === Save pose2d.json ===
        #with open(os.path.join(block_dir, "pose2d.json"), "w") as f:
        #    json.dump(j2d, f)

        # === Save pose3d.json ===
        #with open(os.path.join(block_dir, "pose3d.json"), "w") as f:
        #    json.dump(j3d, f)

        print(f"✓ Salvato blocco in {block_dir}\n")

        block_counter += 1

print(f"\n🏁 Completato! {block_counter} blocchi salvati in totale.")
