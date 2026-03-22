# Use triangolation to calcolate 3D notation in AirPose https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/

import os
import json
import pickle
import numpy as np
import cv2
import pandas as pd
from scipy.signal import savgol_filter
import shutil


CONF_THRESH = 0.25      
REPROJ_THRESH = 70.0    
ROOT_EXP = "/Experiments/angeloa00/AirPose_COCO/train"
MARKER_PATH_TEMPLATE = "/Experiments/angeloa00/AirPose_COCO/{}/markerposes_corrected_all.pkl"
CALIB_PATH_TEMPLATE = "/Experiments/angeloa00/AirPose_COCO/{}/camera_calib.yml"
MACHINE_DIRS = ["machine_1", "machine_2"]
SOURCE_BASE = "/Experiments/angeloa00/AirPose"
TARGET_BASE = "/Experiments/angeloa00/AirPose_COCO"


# =========================================================================
# === INTIATIALIZATION: Copy Machine Folders from AirPose to AirPose_COCO ===
# =========================================================================
print("=== 📂 Initialization: Copying Calibration and Markers files  ===")

for machine_dir_name in MACHINE_DIRS:
    source_path = os.path.join(SOURCE_BASE, machine_dir_name)
    target_path = os.path.join(TARGET_BASE, machine_dir_name)
    
    # 1. Create the destination folder
    os.makedirs(target_path, exist_ok=True)
    
    # 2. Copy calibration and marker files
    calib_file = "camera_calib.yml"
    src_calib = os.path.join(source_path, calib_file)
    dst_calib = os.path.join(target_path, calib_file)
    if os.path.exists(src_calib):
        shutil.copy2(src_calib, dst_calib)
        print(f"[INFO] Copied {calib_file} from {source_path} to {target_path}")
    else:
        print(f"[WARN] File {src_calib} not found. Necessary for triangulation.")
        
    marker_file = "markerposes_corrected_all.pkl"
    src_marker = os.path.join(source_path, marker_file)
    dst_marker = os.path.join(target_path, marker_file)
    if os.path.exists(src_marker):
        shutil.copy2(src_marker, dst_marker)
        print(f"[INFO] Copied {marker_file} from {source_path} to {target_path}")
    else:
        print(f"[WARN] File {src_marker} not found.")

print("=== Copy and Setup complete. Block processing begins. ===")

# =========================================================================

def load_calib(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    D = cv_file.getNode("D").mat()
    cv_file.release()
    return K, D

def get_projection_matrix(marker_entry, K):
    # Extracts R, t from the marker dictionary and constructs P = K @ [R|t]
    cam_key = list(marker_entry.keys())[0]
    rvec = np.array(marker_entry[cam_key]['rvec'])
    tvec = np.array(marker_entry[cam_key]['tvec']).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    # Extrinsics: [R | t]
    Ext = np.hstack([R, tvec])
    P = K @ Ext
    return P, R, tvec

def triangulate_robust(kp1, kp2, K1, D1, K2, D2, P1, P2):
    pts1_2d = np.ascontiguousarray(kp1[:, :2]).astype(np.float32).reshape(-1, 1, 2)
    pts2_2d = np.ascontiguousarray(kp2[:, :2]).astype(np.float32).reshape(-1, 1, 2)
    
    # 1. UNDISTORT POINTS
    # P=K1 (or K2) ensures that the output points are still in pixel coordinates (scaled) rather than normalized coordinates
    pts1_undist = cv2.undistortPoints(pts1_2d, K1, D1, P=K1)
    pts2_undist = cv2.undistortPoints(pts2_2d, K2, D2, P=K2)
    
    # Remove extra dimension given by undistortPoints
    pts1_u = pts1_undist.reshape(-1, 2)
    pts2_u = pts2_undist.reshape(-1, 2)
    
    # 2. VECTORIZED TRIANGULATION
    pts4d = cv2.triangulatePoints(P1, P2, pts1_u.T, pts2_u.T)
    # Converting from 4D homogeneous coordinates to 3D coordinates
    pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3)
    
    # 3. VERIFY REPROJECTION ERROR & CONFIDENCE
    final_pts = []
    
    for i in range(len(pts3d)):
        if kp1[i, 2] < CONF_THRESH or kp2[i, 2] < CONF_THRESH:
            final_pts.append([np.nan]*3)
            continue
            
        X = pts3d[i]
        X_h = np.array([X[0], X[1], X[2], 1.0])
        
        # 
        uv1_proj = P1 @ X_h
        uv1_proj /= uv1_proj[2]
        # The error is calculated with respect to the UNDISTORTED points (pts1_u)
        err1 = np.linalg.norm(uv1_proj[:2] - pts1_u[i])
        
        # Project back to Cam 2
        uv2_proj = P2 @ X_h
        uv2_proj /= uv2_proj[2]
        err2 = np.linalg.norm(uv2_proj[:2] - pts2_u[i])
        
        # Check Reprojection
        if err1 > REPROJ_THRESH or err2 > REPROJ_THRESH:
            final_pts.append([np.nan]*3) # Discard outliers
        else:
            final_pts.append(X.tolist())
            
    return np.array(final_pts) # (N, 3)


# === STATIC DATA LOADING ===
calibs = {}
markers_dict = {}

for m in MACHINE_DIRS:
    K, D = load_calib(CALIB_PATH_TEMPLATE.format(m))
    calibs[m] = (K, D)
    with open(MARKER_PATH_TEMPLATE.format(m), "rb") as f:
        markers_dict[m] = pickle.load(f)


# === PROCESSING BLOCKS ===
blocks = sorted([d for d in os.listdir(ROOT_EXP) if d.startswith("block_")])

for i in range(0, len(blocks)-1, 2):
    b1, b2 = blocks[i], blocks[i+1]
    print(f"Processing pair: {b1} - {b2}")
    
    path1 = os.path.join(ROOT_EXP, b1, "keypoints2d.json")
    path2 = os.path.join(ROOT_EXP, b2, "keypoints2d.json")
    
    with open(path1) as f: kps1_raw = json.load(f)
    with open(path2) as f: kps2_raw = json.load(f)
    
    if isinstance(kps1_raw, list): kps1_raw = {str(idx): v for idx, v in enumerate(kps1_raw)}
    if isinstance(kps2_raw, list): kps2_raw = {str(idx): v for idx, v in enumerate(kps2_raw)}
    
    common_frames = sorted(set(kps1_raw.keys()) & set(kps2_raw.keys()), key=lambda x: int(x))
    
    full_sequence_3d = []
    
    for fid in common_frames:
        # The frame ID for markers is padded with zeros (e.g. "007580")
        fid_marker = f"{int(fid):06d}" 
        
        # Check marker existence
        if fid_marker not in markers_dict["machine_1"] or fid_marker not in markers_dict["machine_2"]:
            full_sequence_3d.append([[np.nan]*3] * 17) # Assuming 17 joints (COCO), fill with NaN
            continue
            
        # Get Matrix P1
        K1, D1 = calibs["machine_1"]
        P1, _, _ = get_projection_matrix(markers_dict["machine_1"][fid_marker], K1)
        
        # Get Matrix P2
        K2, D2 = calibs["machine_2"]
        P2, _, _ = get_projection_matrix(markers_dict["machine_2"][fid_marker], K2)
        
        # Keypoints (N, 3) -> x, y, conf
        pts1 = np.array(kps1_raw[fid]) # shape (N, 3) assumed
        pts2 = np.array(kps2_raw[fid])
        
        # Triangulate current frame
        pts3d_frame = triangulate_robust(pts1, pts2, K1, D1, K2, D2, P1, P2)
        full_sequence_3d.append(pts3d_frame)
    
    # === POST-PROCESSING (Interpolation + Smoothing) ===
    seq_np = np.array(full_sequence_3d) # T x N x 3
    T, N, C = seq_np.shape
    
    cleaned_seq = []
    
    # Iterate for all joints
    for j in range(N):
        traj = pd.DataFrame(seq_np[:, j, :], columns=['x', 'y', 'z'])
        
        # 1. Interpolate NaNs (spline with linear fallback)
        valid_count = traj.dropna().shape[0]
        if valid_count > 4: # If there are enough points for a spline of order 3 (4 minimum)
            try:
                traj_interp = traj.interpolate(method='spline', order=3, limit_direction='both')
            except ValueError as e:
                print(f"[WARN] Spline failed for joint {j} (valid={valid_count}). Falling back to linear interpolation.")
                traj_interp = traj.interpolate(method='linear', limit_direction='both')
        else:
            # Insufficient points, use linear interpolation
            traj_interp = traj.interpolate(method='linear', limit_direction='both')
        
        # 2. Forward Fill and Backward Fill to avoid 0.0 on edges (propagate last/first valid)
        traj_interp = traj_interp.ffill().bfill()
        
        # 3. If there are STILL NaNs left (meaning the WHOLE sequence is NaN), we put 0.0
        traj_interp = traj_interp.fillna(0.0) 
        
        data_to_smooth = traj_interp.values
        
        # 4. Apply Savitzky-Golay only if we have enough data
        if T > 7:
            smoothed_col = np.empty_like(data_to_smooth)
            # Applying the filter on each dimension (x, y, z)
            for c in range(3):
                smoothed_col[:, c] = savgol_filter(data_to_smooth[:, c], window_length=7, polyorder=3)
            cleaned_seq.append(smoothed_col)
        else:
            cleaned_seq.append(data_to_smooth)
            
    # Final reshape to (T, N, 3) -> transposed because the loop was on N
    cleaned_seq = np.array(cleaned_seq).transpose(1, 0, 2)
    
    # Save Output (same file for the two consecutive blocks that form a single sequence)
    out_list = cleaned_seq.tolist()
    
    out1 = os.path.join(ROOT_EXP, b1, "keypoints3d.json")
    out2 = os.path.join(ROOT_EXP, b2, "keypoints3d.json")
    
    with open(out1, "w") as f: json.dump(out_list, f)
    with open(out2, "w") as f: json.dump(out_list, f)
    
    print(f"Saved refined 3D poses for {b1}/{b2}")

print("Computation Completed.")