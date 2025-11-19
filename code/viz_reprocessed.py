import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# --- funzione ausiliaria per disegnare le connessioni 2D ---
def plot_skeleton_2d(ax, joints, pairs, color="lime", radius=2):
    """Disegna i keypoints e i segmenti 2D sul frame."""
    for (i, j) in pairs:
        ax.plot([joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                color=color, linewidth=2)
    ax.scatter(joints[:, 0], joints[:, 1], s=radius * 5, color=color)
    ax.invert_yaxis()  # per coerenza con coordinate immagine


# --- la tua funzione del repo per plottare lo scheletro 3D ---
def plot_skeleton_3d(pose3d, ax, flip_z=False):
    """Funzione presa dal repo (semplificata per esempio)."""
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
            c="orange", lw=2,
        )
    ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], c="red", s=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    ax.grid(False)


# --- funzione principale per creare il video ---
def visualize_element(path_pt, output_path="output_video.mp4", fps=10):
    """Crea un video con a sinistra il video + joints2d e a destra lo scheletro 3D."""
    data = torch.load(path_pt, map_location="cpu")["element"][0]
    video = data["video"].numpy()  # (T, H, W, C)
    joints_2d = data["joints_2d"].numpy()  # (T, N, 2)
    joints_3d = data["joints_3d"].numpy()  # (T, N, 3)

    T = video.shape[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Setup writer
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
            ax1.imshow(frame.astype(np.uint8))
            plot_skeleton_2d(ax1, j2d, pairs=[
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
            ])
            ax1.set_title("Video + 2D joints")
            ax1.axis("off")
            ax1.invert_yaxis()

            # --- scheletro 3D ---
            plot_skeleton_3d(j3d, ax=ax2, flip_z=True)
            ax2.set_title("3D Skeleton (GT)")

            plt.tight_layout()
            writer.grab_frame()

    plt.close(fig)
    print(f"[+] Video salvato in: {output_path}")


# --- esempio d’uso ---
if __name__ == "__main__":
    path_pt = "/mnt/proj2/eu-25-52/ar17/SportsPose_reprocessed/sample_00270_view_back.pt"
    visualize_element(path_pt, output_path="./test.mp4", fps=24)
