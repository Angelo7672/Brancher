# Visualization debug for uav videos

import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path


H36M_PAIRS = [
    (0, 1), (1, 2), (2, 3),        
    (0, 4), (4, 5), (5, 6),        
    (0, 7), (7, 8), (8, 9), (9, 10), 
    (8, 11), (11, 12), (12, 13),   
    (8, 14), (14, 15), (15, 16)    
]


def plot_skeleton_2d(ax, joints, pairs, color="lime"):

    for (i, j) in pairs:
        if joints[i, 2] > 0.1 and joints[j, 2] > 0.1:
            ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                color=color, linewidth=1.5, alpha=0.8
            )
    
    ax.scatter(joints[:, 0], joints[:, 1], s=10, color="red", edgecolors='white', linewidths=0.5)


def plot_bbox(ax, bbox, color="cyan", lw=2):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)


def visualize_training_data(sample_dir, video_path, output_path="debug_viz.mp4", fps=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.array(frames)

    with open(os.path.join(sample_dir, "keypoints2d.json"), "r") as f:
        joints_2d = np.array(json.load(f)) # (T, 17, 3)
    with open(os.path.join(sample_dir, "bboxes.json"), "r") as f:
        bboxes = np.array(json.load(f))    # (T, 4)

    T = min(len(video), len(joints_2d))
    
    writer = FFMpegWriter(fps=fps)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    print(f"Video generation: {output_path} ({T} frames)...")
    
    with writer.saving(fig, output_path, dpi=100):
        for t in range(T):
            ax.clear()
            ax.imshow(video[t])
            
            plot_bbox(ax, bboxes[t], color="yellow")
            plot_skeleton_2d(ax, joints_2d[t], H36M_PAIRS)
            
            ax.set_title(f"Frame {t} - H36M Format")
            ax.axis("off")
            
            writer.grab_frame()
            
    plt.close(fig)
    print(f"Viewing complete!")


if __name__ == "__main__":
    SAMPLE_DIR = "/mnt/proj2/eu-25-52/ar17/DroneDataset/train/video_00026"
    VIDEO_SOURCE = "/mnt/proj2/eu-25-52/ar17/DroneDataset/video_and_flipped/video_0026.mp4"
    
    visualize_training_data(SAMPLE_DIR, VIDEO_SOURCE, output_path="./test_h36m_viz.mp4")