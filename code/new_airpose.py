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

def nonzero_pts_and_mask(pts17):
    mask = ~(np.all(np.isclose(pts17,0.0,atol=1e-8), axis=1))
    return pts17.copy(), mask

def create_video_from_frames(img_dir, start_frame, end_frame):
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    frame_files = frame_files[start_frame:end_frame+1]
    frames = [cv2.imread(os.path.join(img_dir,f)) for f in frame_files]
    return np.stack(frames, axis=0)

def get_common_frame_ids(openpose_data, alphapose_data):
    keys_op = set(map(str, openpose_data.keys()))
    keys_ap = set(map(str, alphapose_data.keys()))
    common = sorted(list(keys_op & keys_ap), key=int)
    return common

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

# --- Frame range personalizzabile ---
START_FRAME = 7580       # 👈 imposta qui il frame di partenza
END_FRAME   = 25561    # 👈 imposta qui il frame finale (inclusivo)
assert (END_FRAME - START_FRAME + 1) % BLOCK_SIZE == 0, \
       "⚠️ L'intervallo deve essere multiplo di 243!"

# === Loop principale ===
block_counter = 0

for global_start in range(START_FRAME, END_FRAME + 1, BLOCK_SIZE):
    global_end = global_start + BLOCK_SIZE - 1
    print(f"\n=== 🔹 Processing block {block_counter} ({global_start}–{global_end}) ===")

    poses_2d_blocks = []
    video_blocks = []
    Ks_list, Rs_list, ts_list = [], [], []

    for machine_dir in machine_dirs:
        with open(os.path.join(machine_dir,"openpose_res.pkl"),"rb") as f:
            openpose_data = pickle.load(f)
        with open(os.path.join(machine_dir,"alphapose_res.json"),"r") as f:
            alphapose_data = json.load(f)
        with open(os.path.join(machine_dir,"markerposes_corrected_all.pkl"),"rb") as f:
            markers = pickle.load(f)
        K, _ = read_calibration(os.path.join(machine_dir,"camera_calib.yml"))

        common_frames = get_common_frame_ids(openpose_data, alphapose_data)
        block_frame_ids = [f for f in common_frames if global_start <= int(f) <= global_end]

        if not block_frame_ids:
            print(f"⚠️ Nessun frame comune trovato in {machine_dir} per intervallo {global_start}-{global_end}")
            continue

        video = create_video_from_frames(os.path.join(machine_dir,"images"), global_start, global_end)
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

        poses_2d = np.stack(poses_2d, axis=0)
        poses_2d = temporal_smooth_and_fill(poses_2d, window=DEFAULT_WINDOW)

        poses_2d_blocks.append(poses_2d)
        Ks_list.append(Ks)
        Rs_list.append(Rs)
        ts_list.append(ts)

    # --- Triangolazione 3D ---
    poses_3d = []
    num_frames = min(poses_2d_blocks[0].shape[0], poses_2d_blocks[1].shape[0])
    print(f"🟢 Processing {num_frames} frames (min length among both views)")

    for fidx in range(num_frames):
        mask_common = ~(np.all(np.isclose(poses_2d_blocks[0][fidx],0.0,atol=1e-8),axis=1)) & \
                      ~(np.all(np.isclose(poses_2d_blocks[1][fidx],0.0,atol=1e-8),axis=1))
        valid_indices = np.where(mask_common)[0]

        pose1_all = poses_2d_blocks[0][fidx][valid_indices].T
        pose2_all = poses_2d_blocks[1][fidx][valid_indices].T

        P1 = Ks_list[0][fidx] @ np.hstack((Rs_list[0][fidx], ts_list[0][fidx]))
        P2 = Ks_list[1][fidx] @ np.hstack((Rs_list[1][fidx], ts_list[1][fidx]))

        if valid_indices.size > 0:
            pts4d_h = cv2.triangulatePoints(P1, P2, pose1_all, pose2_all)
            pts3d = (pts4d_h[:3] / pts4d_h[3]).T
            pts3d_full = np.zeros((17,3), dtype=float)
            pts3d_full[valid_indices] = pts3d
        else:
            pts3d_full = np.zeros((17,3), dtype=float)
        poses_3d.append(pts3d_full)

    poses_3d = np.stack(poses_3d, axis=0)

    # --- Salvataggio del blocco ---
    dataset = []
    for vid, pose2d in zip(video_blocks, poses_2d_blocks):
        dataset.append({
            "video": vid,
            "pose_2d": pose2d,
            "pose_3d": poses_3d
        })

    out_path = f"/Experiments/angeloa00/AirPose/test/dataset_block_{block_counter:03d}.npz"
    np.savez_compressed(out_path, dataset=dataset)
    print(f"✅ Salvato: {out_path} ({len(dataset)} elementi)")

    block_counter += 1

print(f"\n🏁 Completato! {block_counter} blocchi salvati in totale.")
