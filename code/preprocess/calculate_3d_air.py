import os
import json
import pickle
import numpy as np
import cv2

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

# Carichiamo parametri camera da entrambe le machine
Ks, Rs, ts = [], [], []

# Precarichiamo marker poses (assumendo stesso frame_id per iterazione)
marker_data = []
for machine_dir in machine_dirs:
    with open(os.path.join(machine_dir, "markerposes_corrected_all.pkl"), "rb") as f:
        markers = pickle.load(f)
    K, _ = read_calibration(os.path.join(machine_dir, "camera_calib.yml"))
    Ks.append(K)
    marker_data.append(markers)

# === Funzione triangolazione ===
def triangulate_points(p1, p2, K1, R1, t1, K2, R2, t2):
    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])
    pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d

# === Iteriamo sulle coppie block_00i e block_00(i+1) ===
base_exp = "/Experiments/angeloa00/AirPose_COCO/train"
blocks = sorted([d for d in os.listdir(base_exp) if d.startswith("block_")])

for i in range(0, len(blocks)-1, 2):
    b1 = blocks[i]
    b2 = blocks[i+1]

    path1 = os.path.join(base_exp, b1, "keypoints_2d.json")
    path2 = os.path.join(base_exp, b2, "keypoints_2d.json")

    with open(path1) as f: kp1 = json.load(f)
    with open(path2) as f: kp2 = json.load(f)

    frame_ids = sorted(kp1.keys())

    kp3d_out = {}

    for frame_id in frame_ids:
        # Ottieni parametri camera dal marker
        rvec1 = np.array(marker_data[0][frame_id]['0']['rvec'])
        tvec1 = np.array(marker_data[0][frame_id]['0']['tvec']).reshape(3,1)
        R1, _ = cv2.Rodrigues(rvec1)

        rvec2 = np.array(marker_data[1][frame_id]['0']['rvec'])
        tvec2 = np.array(marker_data[1][frame_id]['0']['tvec']).reshape(3,1)
        R2, _ = cv2.Rodrigues(rvec2)

        pts1 = np.array(kp1[frame_id])[:, :2]
        pts2 = np.array(kp2[frame_id])[:, :2]

        pts1_h = pts1.reshape(-1,1,2)
        pts2_h = pts2.reshape(-1,1,2)

        pts3d = []
        for j in range(len(pts1)):
            p1 = pts1[j]
            p2 = pts2[j]
            p1 = np.array([[p1[0]], [p1[1]]], dtype=np.float32)
            p2 = np.array([[p2[0]], [p2[1]]], dtype=np.float32)

            P = triangulate_points(p1, p2, Ks[0], R1, tvec1, Ks[1], R2, tvec2)
            pts3d.append(P[0].tolist())

        kp3d_out[frame_id] = pts3d

    out1 = os.path.join(base_exp, b1, "keypoints_3d.json")
    out2 = os.path.join(base_exp, b2, "keypoints_3d.json")

    with open(out1, "w") as f: json.dump(kp3d_out, f)
    with open(out2, "w") as f: json.dump(kp3d_out, f)

print("Triangolazione completata.")
