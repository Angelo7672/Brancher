import os
import json
import cv2
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
def plot_skeleton_3d(pose3d, ax, flip_z=False):
    skeleton_parents = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [5, 3],
        [6, 4],
        [3, 1],
        [4, 2],
        [1, 0],
        [2, 0],
    ]
    if flip_z:
        pose3d[..., 2] *= -1

    for (i, j) in skeleton_parents:
        ax.plot(
            [pose3d[i, 0], pose3d[j, 0]],
            [pose3d[i, 1], pose3d[j, 1]],
            [pose3d[i, 2], pose3d[j, 2]],
            c="orange",
            lw=2,
        )
    ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], c="red", s=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    ax.grid(False)


# --- funzione principale aggiornata ---
def visualize_element_folder(sample_dir, output_path="output_video.mp4", fps=10):
    """
    Crea un video con a sinistra il video + joints2d e a destra lo scheletro 3D.
    sample_dir: path della cartella che contiene video.mp4, joints_2d.json, joints_3d.json
    """
    # --- carica video ---
    video_path = os.path.join(sample_dir, "video.mp4")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # resta in RGB
        frames.append(frame)
    cap.release()
    video = np.stack(frames, axis=0)  # (T, H, W, C)

    # --- carica annotazioni ---
    with open(os.path.join(sample_dir, "pose2d.json"), "r") as f:
        joints_2d = np.array(json.load(f))
    with open(os.path.join(sample_dir, "pose3d.json"), "r") as f:
        joints_3d = np.array(json.load(f))

    T = min(len(video), len(joints_2d), len(joints_3d))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- setup writer ---
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(10, 5))

    with writer.saving(fig, output_path, dpi=100):
        for t in range(T):
            frame = video[t]
            j2d = joints_2d[t]
            j3d = joints_3d[t].copy()
            j3d[..., 2] *= -1  # flip lungo Z

            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            # --- frame 2D con joints ---
            ax1.imshow(frame)
            plot_skeleton_2d(ax1, j2d, pairs=[
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                [7, 9], [8, 10], [5, 3], [6, 4], [3, 1],
                [4, 2], [1, 0], [2, 0],
            ])
            ax1.set_title("Video + 2D joints")
            ax1.axis("off")

            # --- scheletro 3D ---
            plot_skeleton_3d(j3d, ax=ax2, flip_z=False)
            ax2.set_title("3D Skeleton (GT)")

            plt.tight_layout()
            writer.grab_frame()

    plt.close(fig)
    print(f"[+] Video salvato in: {output_path}")


# --- esempio d'uso ---
if __name__ == "__main__":
    #sample_dir = "/mnt/proj2/eu-25-52/ar17/SportsPose_reprocessed_v2/sample_00270_view_back"
    sample_dir = "/Experiments/angeloa00/AirPose/test/block_061"
    visualize_element_folder(sample_dir, output_path="./debug_output.mp4", fps=24)