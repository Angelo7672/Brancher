import os
import cv2
import json
import pickle
import numpy as np

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
    pose_op = np.array(openpose_data[frame_id]['pose'][0])[:, :2]
    ap_people = alphapose_data[frame_id].get('people', [])
    if len(ap_people) == 0:
        raise RuntimeError(f"AlphaPose: nessuna persona in frame {frame_id}")
    raw_ap = np.array(ap_people[0].get('pose_keypoints_2d', ap_people[0].get('keypoints', [])))
    pose_ap = raw_ap.reshape(-1, 3)[:, :2]

    # Mapping OpenPose(25) → COCO17
    op_to_coco = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11], dtype=int)
    pose_op_common = pose_op[op_to_coco]

    # Mapping AlphaPose18 → COCO17 (escludendo neck)
    ap_to_coco = np.array([0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10], dtype=int)
    pose_ap_common = pose_ap[ap_to_coco]

    # Media robusta frame-by-frame
    pose_avg = np.zeros_like(pose_op_common)
    for i in range(17):
        op, ap = pose_op_common[i], pose_ap_common[i]
        if np.allclose(op,0) and np.allclose(ap,0):
            pose_avg[i] = [0,0]  # temporaneo, verrà interpolato dopo
        elif np.allclose(op,0):
            pose_avg[i] = ap
        elif np.allclose(ap,0):
            pose_avg[i] = op
        else:
            pose_avg[i] = (op + ap) / 2.0
    return pose_avg

def nonzero_pts_and_mask(pts17):
    mask = ~(np.all(np.isclose(pts17,0.0,atol=1e-8), axis=1))
    return pts17.copy(), mask

def create_video_from_frames(img_dir, start_frame, end_frame):
    frame_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    frame_files.sort()
    frame_files = frame_files[start_frame:end_frame+1]
    frames = [cv2.imread(os.path.join(img_dir,f)) for f in frame_files]
    return np.stack(frames, axis=0)

def get_common_frame_ids(openpose_data, alphapose_data):
    keys_op = set(map(str, openpose_data.keys()))
    keys_ap = set(map(str, alphapose_data.keys()))
    common = sorted(list(keys_op & keys_ap), key=int)
    return common

# --- Funzione originale: interpolazione + moving average (manteniamo questa) ---
def temporal_smooth_and_fill(poses_2d_seq, window=5):
    """
    Interpola i 0 e applica moving-average per smussare.
    Input: poses_2d_seq shape (T,17,2)
    """
    smoothed = poses_2d_seq.copy().astype(float)
    T = smoothed.shape[0]
    half = window // 2
    for k in range(17):
        for dim in range(2):
            coords = smoothed[:, k, dim]
            mask_valid = coords != 0
            if np.any(mask_valid):
                # interpola i valori mancanti
                coords[~mask_valid] = np.interp(
                    np.flatnonzero(~mask_valid),
                    np.flatnonzero(mask_valid),
                    coords[mask_valid]
                )
            else:
                # se tutti mancanti, lascialo a zero (non ideale ma raro)
                coords[:] = 0
            # moving average (media semplice)
            kernel = np.ones(window) / window
            coords_smooth = np.convolve(coords, kernel, mode='same')
            smoothed[:, k, dim] = coords_smooth
    return smoothed

# === Parametri ===
block_size = 243
dataset = []
# default window per moving-average
DEFAULT_WINDOW = 5

for block_idx in range(2):
    start_frame = block_idx * block_size
    end_frame = start_frame + block_size - 1

    poses_2d_blocks = []
    video_blocks = []
    Ks_list, Rs_list, ts_list = [], [], []

    for machine_dir in machine_dirs:
        # load
        with open(os.path.join(machine_dir,"openpose_res.pkl"),"rb") as f:
            openpose_data = pickle.load(f)
        with open(os.path.join(machine_dir,"alphapose_res.json"),"r") as f:
            alphapose_data = json.load(f)
        with open(os.path.join(machine_dir,"markerposes_corrected_all.pkl"),"rb") as f:
            markers = pickle.load(f)
        K, _ = read_calibration(os.path.join(machine_dir,"camera_calib.yml"))

        common_frames = get_common_frame_ids(openpose_data, alphapose_data)
        block_frame_ids = common_frames[start_frame:end_frame+1]

        video = create_video_from_frames(os.path.join(machine_dir,"images"), start_frame, end_frame)
        video_blocks.append(video)

        poses_2d = []
        Ks, Rs, ts = [], [], []

        for fidx in block_frame_ids:
            frame_id = str(fidx)
            fused = fuse_keypoints(openpose_data, alphapose_data, frame_id)
            pts17, mask17 = nonzero_pts_and_mask(fused)
            poses_2d.append(pts17)

            Ks.append(K)
            rvec = np.array(markers[frame_id]['0']['rvec'])
            tvec = np.array(markers[frame_id]['0']['tvec']).reshape(3,)
            R, _ = cv2.Rodrigues(rvec)
            Rs.append(R)
            ts.append(tvec.reshape(3,1))

        poses_2d = np.stack(poses_2d, axis=0)  # shape (T,17,2)

        # --- 1) Interpolazione + moving-average (original filter you had) ---
        poses_2d = temporal_smooth_and_fill(poses_2d, window=DEFAULT_WINDOW)

        poses_2d_blocks.append(poses_2d)
        Ks_list.append(Ks)
        Rs_list.append(Rs)
        ts_list.append(ts)

    # --- Triangolazione 3D (uguale a prima) ---
    poses_3d = []
    num_frames = poses_2d_blocks[0].shape[0]
    for fidx in range(num_frames):
        mask_common = ~(np.all(np.isclose(poses_2d_blocks[0][fidx],0.0,atol=1e-8),axis=1)) & \
                      ~(np.all(np.isclose(poses_2d_blocks[1][fidx],0.0,atol=1e-8),axis=1))
        valid_indices = np.where(mask_common)[0]

        pose1_all = poses_2d_blocks[0][fidx][valid_indices].T
        pose2_all = poses_2d_blocks[1][fidx][valid_indices].T

        P1 = Ks_list[0][fidx] @ np.hstack((Rs_list[0][fidx], ts_list[0][fidx]))
        P2 = Ks_list[1][fidx] @ np.hstack((Rs_list[1][fidx], ts_list[1][fidx]))

        if valid_indices.size>0:
            pts4d_h = cv2.triangulatePoints(P1,P2,pose1_all,pose2_all)
            pts3d = (pts4d_h[:3]/pts4d_h[3]).T
            pts3d_full = np.zeros((17,3),dtype=float)
            pts3d_full[valid_indices] = pts3d
        else:
            pts3d_full = np.zeros((17,3),dtype=float)

        poses_3d.append(pts3d_full)

    poses_3d = np.stack(poses_3d, axis=0)

    for vid, pose2d in zip(video_blocks, poses_2d_blocks):
        dataset.append({
            "video": vid,
            "pose_2d": pose2d,
            "pose_3d": poses_3d
        })

# Salvataggio finale
np.savez_compressed("dataset_airpose.npz", dataset=dataset)
print(f"✅ Dataset pronto: {len(dataset)} elementi (2 blocchi x 2 macchine)")



