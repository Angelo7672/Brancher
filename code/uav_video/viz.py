import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path

# --- Connessioni Scheletro H36M (17 joints) ---
H36M_PAIRS = [
    (0, 1), (1, 2), (2, 3),        # Gamba destra
    (0, 4), (4, 5), (5, 6),        # Gamba sinistra
    (0, 7), (7, 8), (8, 9), (9, 10), # Tronco e testa
    (8, 11), (11, 12), (12, 13),   # Braccio sinistro
    (8, 14), (14, 15), (15, 16)    # Braccio destro
]

def plot_skeleton_2d(ax, joints, pairs, color="lime"):
    """Disegna keypoints e segmenti 2D."""
    # joints shape: (17, 3) -> [x, y, conf]
    for (i, j) in pairs:
        if joints[i, 2] > 0.1 and joints[j, 2] > 0.1: # Disegna solo se visibili
            ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                color=color, linewidth=1.5, alpha=0.8
            )
    
    # Scatter per i giunti (usa la confidenza per la dimensione)
    ax.scatter(joints[:, 0], joints[:, 1], s=10, color="red", edgecolors='white', linewidths=0.5)

def plot_bbox(ax, bbox, color="cyan", lw=2):
    """Disegna il rettangolo del bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)

def visualize_training_data(sample_dir, video_path, output_path="debug_viz.mp4", fps=30):
    """
    Visualizza video originale con overlay di BBox e scheletro H36M.
    """
    # --- 1. Carica Video ---
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.array(frames)

    # --- 2. Carica Annotazioni ---
    with open(os.path.join(sample_dir, "keypoints2d.json"), "r") as f:
        joints_2d = np.array(json.load(f)) # (T, 17, 3)
    with open(os.path.join(sample_dir, "bboxes.json"), "r") as f:
        bboxes = np.array(json.load(f))    # (T, 4)

    T = min(len(video), len(joints_2d))
    
    # --- 3. Setup Video Writer ---
    writer = FFMpegWriter(fps=fps)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    print(f"Generazione video: {output_path} ({T} frames)...")
    
    with writer.saving(fig, output_path, dpi=100):
        for t in range(T):
            ax.clear()
            
            # Mostra frame
            ax.imshow(video[t])
            
            # Disegna Bounding Box
            plot_bbox(ax, bboxes[t], color="yellow")
            
            # Disegna Scheletro H36M
            plot_skeleton_2d(ax, joints_2d[t], H36M_PAIRS)
            
            ax.set_title(f"Frame {t} - H36M Format")
            ax.axis("off")
            
            writer.grab_frame()
            
    plt.close(fig)
    print(f"Visualizzazione completata!")

if __name__ == "__main__":
    # ESEMPIO: Punta alla cartella del blocco elaborato e al video originale
    # Se hai usato lo script precedente, il video originale è quello a 30fps
    SAMPLE_DIR = "/mnt/proj2/eu-25-52/ar17/DroneDataset/train/video_00715"
    
    # Attenzione: qui devi passare il video corrispondente a quel blocco specifico
    # Se il blocco 00000 è il primo chunk del video_0000.mp4:
    VIDEO_SOURCE = "/mnt/proj2/eu-25-52/ar17/DroneDataset/video_and_flipped/video_0715.mp4"
    
    visualize_training_data(SAMPLE_DIR, VIDEO_SOURCE, output_path="./test_h36m_viz.mp4")