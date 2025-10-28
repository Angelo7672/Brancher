import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Percorsi ===
openpose_path = "/DATA/AirPose/copenet_dji_real_data/machine_1/openpose_res.pkl"
alphapose_path = "/DATA/AirPose/copenet_dji_real_data/machine_1/alphapose_res.json"

# === Caricamento ===
print("Caricamento dati...")
with open(openpose_path, "rb") as f:
    openpose_data = pickle.load(f)
with open(alphapose_path, "r") as f:
    alphapose_data = json.load(f)

# === Seleziona primo frame comune ===
common_frames = sorted(set(openpose_data.keys()) & set(alphapose_data.keys()))
if not common_frames:
    raise RuntimeError("Nessun frame comune trovato.")
frame_id = common_frames[0]
print("Usando frame:", frame_id)

# === Estrai keypoints ===
pose_op = np.array(openpose_data[frame_id]['pose'][0])[:, :2]  # (25,2)
ap_people = alphapose_data[frame_id].get('people', [])
if len(ap_people) == 0:
    raise RuntimeError(f"AlphaPose: nessuna persona trovata in frame {frame_id}")
raw_ap = np.array(ap_people[0].get('pose_keypoints_2d', ap_people[0].get('keypoints', [])))
if raw_ap.size == 0:
    raise RuntimeError("AlphaPose: keypoints non trovati nella struttura people[0].")
pose_ap = raw_ap.reshape(-1, 3)[:, :2]  # (18,2) tipicamente

print(f"OpenPose keypoints: {pose_op.shape[0]}")
print(f"AlphaPose keypoints: {pose_ap.shape[0]}")

# === AlphaPose 18 → COCO 17 ===
# Layout AlphaPose18 (MPII-like):
# 0:nose, 1:neck, 2:Rsho, 3:Relb, 4:Rwr, 5:Lsho, 6:Lelb, 7:Lwr,
# 8:Rhip, 9:Rkne, 10:Rank, 11:Lhip, 12:Lkne, 13:Rank,
# 14:Reye, 15:Leye, 16:Rear, 17:Lear

# COCO17 layout: 0:nose, 1:LEye, 2:REye, 3:LEar, 4:REar,
# 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist,
# 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee,
# 15:LAnkle, 16:RAnkle

# Mapping AlphaPose18 -> COCO17 (escludendo neck)
ap_to_coco = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
pose_ap_common = pose_ap[ap_to_coco]

# === OpenPose (25) → COCO (17) ===
op_to_coco = np.array([0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11], dtype=int)
pose_op_common = pose_op[op_to_coco]

assert pose_op_common.shape == pose_ap_common.shape == (17, 2)

# === Calcolo media con regole sui (0,0) ===
pose_avg = np.zeros_like(pose_op_common)
for i in range(17):
    op, ap = pose_op_common[i], pose_ap_common[i]
    is_op_zero = np.allclose(op, 0.0)
    is_ap_zero = np.allclose(ap, 0.0)
    if is_op_zero and is_ap_zero:
        pose_avg[i] = [0, 0]
    elif is_op_zero:
        pose_avg[i] = ap
    elif is_ap_zero:
        pose_avg[i] = op
    else:
        pose_avg[i] = (op + ap) / 2

# === Funzioni di supporto ===
def nonzero_points_with_idx(pts):
    mask = ~(np.all(np.isclose(pts, 0.0, atol=1e-8), axis=1))
    return np.where(mask)[0], pts[mask]

def plot_points(pts, idxs, title, color, out_path):
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=18)
    plt.scatter(pts[:, 0], pts[:, 1], c=color, s=70, marker='o')
    for i, (x, y) in zip(idxs, pts):
        plt.text(x + 3, y + 3, str(i), color=color, fontsize=14, weight='bold')
    plt.gca().invert_yaxis()
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Salvato: {out_path}")

# === Plot separati ===
op_idx, op_pts = nonzero_points_with_idx(pose_op_common)
ap_idx, ap_pts = nonzero_points_with_idx(pose_ap_common)
avg_idx, avg_pts = nonzero_points_with_idx(pose_avg)

plot_points(op_pts, op_idx, f"OpenPose Keypoints - Frame {frame_id}", "blue", "openpose_frame.png")
plot_points(ap_pts, ap_idx, f"AlphaPose Keypoints (mapped→COCO) - Frame {frame_id}", "red", "alphapose_frame.png")
plot_points(avg_pts, avg_idx, f"Media Keypoints - Frame {frame_id}", "green", "merged_pose_frame.png")

print("\nCompletato ✅")





