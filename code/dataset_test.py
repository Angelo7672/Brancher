import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# === Percorsi ===
dataset_path = "/Experiments/angeloa00/AirPose/train/dataset_block_030.npz"
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# === Parametri video ===
fps = 25

# === Colori keypoints per plot ===
colors = plt.cm.tab20(np.linspace(0,1,17))

# === Carica dataset ===
data = np.load(dataset_path, allow_pickle=True)
dataset = data['dataset']

for idx, element in enumerate(dataset):
    video_array = element['video']         # (num_frames, H, W, 3)
    pose_2d = element['pose_2d']           # (num_frames, 17, 2)
    pose_3d = element['pose_3d']           # (num_frames, 17, 3)
    num_frames, H, W, _ = video_array.shape
    num_pose_2d = pose_2d.shape[0]
    num_pose_3d = pose_3d.shape[0]

    # --- 1) Salva video originale ---
    out_path_video = os.path.join(output_dir, f"video_{idx+1}_original.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path_video, fourcc, fps, (W,H))
    for f in range(num_frames):
        frame = video_array[f]
        #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2]==3 else frame
        out.write(frame)
    out.release()
    print(f"✅ Video originale {idx+1} salvato.")

    # --- 2) Video pose 2D ---
    out_path_2d = os.path.join(output_dir, f"video_{idx+1}_pose2d.mp4")
    out2d = cv2.VideoWriter(out_path_2d, fourcc, fps, (W,H))

    for f in range(num_pose_2d):
        plt.figure(figsize=(W/100, H/100), dpi=100)
        plt.xlim(0,W)
        plt.ylim(H,0)  # y invertita
        plt.axis('off')
        pts = pose_2d[f]
        for k in range(17):
            plt.plot(pts[k,0], pts[k,1], 'o', color=colors[k])
        plt.tight_layout()
        plt.draw()
        plt.canvas = plt.get_current_fig_manager().canvas
        plt.canvas.draw()
        img = np.frombuffer(plt.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(plt.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (W,H))
        out2d.write(img_bgr)
        plt.close()
    out2d.release()
    print(f"✅ Video pose 2D {idx+1} salvato.")

    # --- 3) Video pose 3D (senza occhi) ---
    out_path_3d = os.path.join(output_dir, f"video_{idx+1}_pose3d.mp4")
    out3d = cv2.VideoWriter(out_path_3d, fourcc, fps, (W, H))

    # Connessioni COCO17 (senza quelle che coinvolgono gli occhi)
    COCO_PAIRS = [
        # (0,1), (0,2),          # naso → occhi (rimosse)
        # (1,3), (2,4),          # occhi → orecchie (rimosse)
        (5,6),                  # spalle
        (5,7), (7,9),           # braccio sinistro
        (6,8), (8,10),          # braccio destro
        (11,12),                # anche
        (5,11), (6,12),         # colonna
        (11,13), (13,15),       # gamba sinistra
        (12,14), (14,16)        # gamba destra
    ]

    # Keypoints da disegnare (tutti tranne occhi: 1, 2)
    EXCLUDE_KPTS = {1, 2}

    for f in range(num_pose_3d):
        pts = pose_3d[f]

        if np.allclose(pts, 0):
            continue

        # Limiti dinamici con margine
        min_vals = np.nanmin(pts, axis=0)
        max_vals = np.nanmax(pts, axis=0)
        padding = (max_vals - min_vals) * 0.2
        min_vals -= padding
        max_vals += padding

        fig = plt.figure(figsize=(W/100, H/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Imposta limiti e assi
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=15, azim=-70)
        ax.grid(True)

        # Disegna keypoints (escludendo occhi)
        for k in range(17):
            if k not in EXCLUDE_KPTS and not np.allclose(pts[k], 0):
                ax.scatter(pts[k, 0], pts[k, 1], pts[k, 2],
                           color=colors[k], s=40, edgecolor='k')

        # Disegna connessioni (scheletro)
        for (i, j) in COCO_PAIRS:
            if not (np.allclose(pts[i], 0) or np.allclose(pts[j], 0)):
                xs, ys, zs = zip(pts[i], pts[j])
                ax.plot(xs, ys, zs, color='black', linewidth=2, alpha=0.8)

        plt.tight_layout()
        plt.canvas = plt.get_current_fig_manager().canvas
        plt.canvas.draw()
        img = np.frombuffer(plt.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(plt.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (W, H))
        out3d.write(img_bgr)
        plt.close(fig)

    out3d.release()
    print(f"✅ Video pose 3D {idx+1} salvato (senza occhi).")

print("🎉 Tutti i 6 video creati!")
