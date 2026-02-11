import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# --- funzione ausiliaria per disegnare le connessioni 2D ---
def plot_skeleton_2d(ax, joints, pairs, color="lime", radius=2):
    """Disegna i keypoints e i segmenti 2D sul frame."""
    for (i, j) in pairs:
        ax.plot(
            [joints[i, 0], joints[j, 0]],
            [joints[i, 1], joints[j, 1]],
            color=color,
            linewidth=2,
        )
    ax.scatter(joints[:, 0], joints[:, 1], s=radius * 5, color=color)
    ax.invert_yaxis()  # per coerenza con coordinate immagine


# --- funzione per plottare lo scheletro 3D ---
def plot_skeleton_3d(pose3d, ax):
    skeleton_parents = [
        [15, 13],[13, 11],[16, 14],[14, 12],[11, 12],
        [5, 11],[6, 12],[5, 6],[5, 7],[6, 8],
        [7, 9],[8, 10],[5, 3],[6, 4],[3, 1],
        [4, 2],[1, 0],[2, 0],
    ]
    pose = pose3d.copy()

    for (i, j) in skeleton_parents:
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            [pose[i, 2], pose[j, 2]],
            c="orange",
            lw=2,
        )
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c="red", s=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    ax.grid(False)

def plot_bbox(ax, bbox, color="cyan", lw=2):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    rect_x = [x1, x2, x2, x1, x1]
    rect_y = [y1, y1, y2, y2, y1]
    ax.plot(rect_x, rect_y, color=color, linewidth=lw)


# --- funzione principale aggiornata ---
def visualize_element_folder(sample_dir, output_path="output_video.mp4", fps=10):
    """
    Crea un video con a sinistra il video + joints2d e a destra lo scheletro 3D.
    sample_dir: path della cartella che contiene video.mp4, joints_2d.json, joints_3d.json
    """
    # --- carica video ---
<<<<<<< HEAD
    video_path = os.path.join("/Experiments/angeloa00/AirPose/test/block_010", "video.mp4")
=======
    #video_path = os.path.join("/Experiments/angeloa00/AirPose/train/block_010", "video.mp4")
    video_path = os.path.join("/mnt/proj2/eu-25-52/ar17/SportsPose_reprocessed_v2/sample_00372_view_back-left", "video.mp4")
>>>>>>> 38168cab7abbbbde15c4e47c5195490322339dac
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # resta in RGB
        frame = cv2.flip(frame, 0)  # flip verticale
        frames.append(frame)
    cap.release()
    video = np.stack(frames, axis=0)  # (T, H, W, C)

    # --- carica annotazioni ---
    #with open(os.path.join(sample_dir, "keypoints2d.json"), "r") as f:
    #    joints_2d = np.array(json.load(f))
    #with open(os.path.join(sample_dir, "keypoints3d.json"), "r") as f:
    #    joints_3d = np.array(json.load(f))
    #with open(os.path.join(sample_dir, "bboxes.json"), "r") as f:
    #    bboxes = np.array(json.load(f))  # (T, 4)
    with open(os.path.join(sample_dir, "pose2d.json"), "r") as f:
        joints_2d = np.array(json.load(f))
    with open(os.path.join(sample_dir, "pose3d.json"), "r") as f:
        joints_3d = np.array(json.load(f))
<<<<<<< HEAD
    with open(os.path.join(sample_dir, "bboxes.json"), "r") as f:
        bboxes = np.array(json.load(f))  # (T, 4)
=======
>>>>>>> 38168cab7abbbbde15c4e47c5195490322339dac

    T = min(len(video), len(joints_2d), len(joints_3d))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- setup writer ---
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(10, 5))
    frame_height, frame_width = video[0].shape[0:2]

    with writer.saving(fig, output_path, dpi=100):
        for t in range(T):
            frame = video[t]

            j2d_local = joints_2d[t]      # (J,3) nel crop
            bbox = bboxes[t]              # [x1,y1,x2,y2]
            x1, y1, x2, y2 = bbox
            bbox = np.array([x1, frame_height - y2, x2, frame_height - y1])  # flip verticale
            j2d = j2d_local.copy()

            j3d = joints_3d[t].copy()
            j3d[..., 1] *= -1  # flip lungo Y
            j3d[..., 2] *= -1  # flip lungo Z

            # --- flip keypoints 2D verticale ---
            j2d[:, 1] = frame.shape[0] - j2d[:, 1]

            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            # --- frame 2D con joints e bounding box ---
            ax1.imshow(frame)
            #plot_bbox(ax1, bbox)
            plot_skeleton_2d(ax1, j2d, pairs=[
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                [7, 9], [8, 10], [5, 3], [6, 4], [3, 1],
                [4, 2], [1, 0], [2, 0],
            ])
            ax1.set_title("Video + 2D joints")
            ax1.axis("off")

            # --- scheletro 3D ---
            plot_skeleton_3d(j3d, ax=ax2)
            ax2.set_title("3D Skeleton (GT)")

            plt.tight_layout()
            writer.grab_frame()

    plt.close(fig)
    print(f"[+] Video salvato in: {output_path}")


# --- esempio d'uso ---
if __name__ == "__main__":
    #sample_dir = "/Experiments/angeloa00/SportsPose_reprocessed_v3/sample_00372_view_back-left"
    sample_dir = "/Experiments/angeloa00/AirPose_COCO/test/block_010"
    visualize_element_folder(sample_dir, output_path="./debug_output.mp4", fps=24)