import os
import json
import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === Percorsi base ===
root = "/DATA/AirPose/copenet_dji_real_data"
machine_dirs = [os.path.join(root, "machine_1"), os.path.join(root, "machine_2")]

# === Funzione per leggere calibrazione ===
def read_calibration(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    D = cv_file.getNode("D").mat()
    cv_file.release()
    return K, D

# === Funzione: mappatura + media (OpenPose + AlphaPose → COCO17) ===
def fuse_keypoints(openpose_data, alphapose_data, frame_id):
    # OpenPose 25 -> (25,2)
    pose_op = np.array(openpose_data[frame_id]['pose'][0])[:, :2]

    # AlphaPose (18) raw
    ap_people = alphapose_data[frame_id].get('people', [])
    if len(ap_people) == 0:
        raise RuntimeError(f"AlphaPose: nessuna persona in frame {frame_id}")
    raw_ap = np.array(ap_people[0].get('pose_keypoints_2d', ap_people[0].get('keypoints', [])))
    if raw_ap.size == 0:
        raise RuntimeError("AlphaPose: keypoints non trovati.")
    pose_ap = raw_ap.reshape(-1, 3)[:, :2]  # (18,2) tipicamente

    # --- Mappatura OpenPose(25) -> COCO17 ---
    op_to_coco = np.array([0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11], dtype=int)
    pose_op_common = pose_op[op_to_coco]

    # --- Mappatura AlphaPose18 -> COCO17 (escludendo neck index 1) ---
    # ap layout assumed 18 with neck at index 1
    ap_to_coco = np.array([0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10], dtype=int)
    if pose_ap.shape[0] == 18:
        # use mapping to reorder & pick the 17 COCO points
        pose_ap_common = pose_ap[ap_to_coco]
    else:
        # fallback: if not 18, try first-17, or pad
        if pose_ap.shape[0] >= 17:
            pose_ap_common = pose_ap[:17]
        else:
            pose_ap_common = np.zeros((17,2), dtype=float)
            pose_ap_common[:pose_ap.shape[0]] = pose_ap

    assert pose_op_common.shape == pose_ap_common.shape == (17, 2)
    # --- Media robusta ---
    pose_avg = np.zeros_like(pose_op_common)
    for i in range(17):
        op = pose_op_common[i]
        ap = pose_ap_common[i]
        is_op_zero = np.allclose(op, 0.0)
        is_ap_zero = np.allclose(ap, 0.0)
        if is_op_zero and is_ap_zero:
            pose_avg[i] = [0, 0]
        elif is_op_zero:
            pose_avg[i] = ap
        elif is_ap_zero:
            pose_avg[i] = op
        else:
            pose_avg[i] = (op + ap) / 2.0
    return pose_avg

# === Funzione per ottenere punti non-zero e mask ===
def nonzero_pts_and_mask(pts17):
    mask = ~(np.all(np.isclose(pts17, 0.0, atol=1e-8), axis=1))  # shape (17,)
    pts = pts17.copy()
    return pts, mask

# === Inizio elaborazione per le due macchine ===
poses_2d_full = []   # list of (17,2) arrays (fused per camera)
masks_list = []
Ks, Rs, ts = [], [], []
frame_id = None

for machine_dir in machine_dirs:
    print(f"\n📂 Elaboro {machine_dir}")

    with open(os.path.join(machine_dir, "openpose_res.pkl"), "rb") as f:
        openpose_data = pickle.load(f)
    with open(os.path.join(machine_dir, "alphapose_res.json"), "r") as f:
        alphapose_data = json.load(f)
    with open(os.path.join(machine_dir, "markerposes_corrected_all.pkl"), "rb") as f:
        markers = pickle.load(f)
    K, _ = read_calibration(os.path.join(machine_dir, "camera_calib.yml"))

    if frame_id is None:
        # trova primo frame comune
        common = sorted(set(openpose_data.keys()) & set(alphapose_data.keys()))
        if not common:
            raise RuntimeError("Nessun frame comune")
        frame_id = common[0]

    fused = fuse_keypoints(openpose_data, alphapose_data, frame_id)  # (17,2)
    pts17, mask17 = nonzero_pts_and_mask(fused)
    poses_2d_full.append(pts17)   # keep full 17 (zeros kept)
    masks_list.append(mask17)
    Ks.append(K)

    # extrinsics
    rvec = np.array(markers[frame_id]['0']['rvec'])
    tvec = np.array(markers[frame_id]['0']['tvec']).reshape(3, )
    R, _ = cv2.Rodrigues(rvec)
    Rs.append(R)
    ts.append(tvec.reshape(3,1))

print(f"\nFrame comune usato: {frame_id}")

# === Calcola mask comune: solo punti validi in entrambe le viste ===
mask_common = masks_list[0] & masks_list[1]
valid_indices = np.where(mask_common)[0]
if valid_indices.size == 0:
    raise RuntimeError("Nessun keypoint valido in entrambe le viste per triangolare.")

print(f"Keypoints validi (indici COCO): {valid_indices.tolist()}")

# === Prepara punti per triangolazione: shape (2, N) per ciascuna vista ===
# For cv2.triangulatePoints we need 2xN arrays in pixel coords.
pose1_all = poses_2d_full[0][valid_indices].T  # shape (2, N)
pose2_all = poses_2d_full[1][valid_indices].T  # shape (2, N)

# === Matrici di proiezione (3x4) ===
P1 = Ks[0] @ np.hstack((Rs[0], ts[0]))
P2 = Ks[1] @ np.hstack((Rs[1], ts[1]))

# === Triangolazione ===
pts4d_h = cv2.triangulatePoints(P1, P2, pose1_all, pose2_all)  # 4xN
pts3d = (pts4d_h[:3] / pts4d_h[3]).T  # Nx3
print(f"✅ Ricostruiti {pts3d.shape[0]} punti 3D (per indici COCO: {valid_indices.tolist()})")

# === Salva risultato 3D (con indice COCO associato) ===
out3d = {"frame": frame_id, "indices": valid_indices.tolist(), "points3d": pts3d}
#with open("triangulated_3d.npy", "wb") as f:
#    np.save(f, out3d)
#print("Salvato triangulated_3d.npy")

# === Plot 2D debug cartesiani per ciascuna vista (solo punti validi) ===
for i, machine_dir in enumerate(machine_dirs):
    pts = poses_2d_full[i][valid_indices]  # (M,2)
    plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0], pts[:,1], c='C0', s=60)
    for j, (x,y) in zip(valid_indices, pts):
        plt.text(x, y, str(j), fontsize=14, weight='bold', color='C0')
    plt.gca().invert_yaxis()
    plt.title(f"Debug 2D - {os.path.basename(machine_dir)} - frame {frame_id}", fontsize=14)
    plt.xlabel("x [px]"); plt.ylabel("y [px]"); plt.grid(True)
    outname = f"debug_2d_machine{i+1}.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()
    print(f"Salvato: {outname}")

# === Plot 3D ===
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], c='red', s=40)
for idx_pair, p in zip(valid_indices.tolist(), pts3d):
    ax.text(p[0], p[1], p[2], str(idx_pair), color='black')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f'Triangolazione 3D - Frame {frame_id}')
ax.view_init(elev=20, azim=-70)
plt.tight_layout()
plt.savefig("pose3d_fused_debug.png", dpi=300)
print("Salvato: pose3d_fused_debug.png")







