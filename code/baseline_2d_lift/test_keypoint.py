import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from keypoint_extractor import KeypointExtractorSeq

# === COCO skeleton ===
COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def draw_skeleton(frame, keypoints, H, W, conf_thresh=0.0):
    """
    Disegna i keypoint e lo scheletro sul frame.
    I keypoints sono normalizzati in [0,1], quindi li riportiamo in pixel.
    """
    for j, (x, y, c) in enumerate(keypoints):
        if c > conf_thresh and 0 <= x <= 1 and 0 <= y <= 1:
            cx, cy = int(x * W), int(y * H)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(frame, str(j), (cx + 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    for (j1, j2) in COCO_PAIRS:
        if (keypoints[j1][2] > conf_thresh and keypoints[j2][2] > conf_thresh):
            x1, y1 = int(keypoints[j1][0] * W), int(keypoints[j1][1] * H)
            x2, y2 = int(keypoints[j2][0] * W), int(keypoints[j2][1] * H)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


def process_video(video_path, out_skeleton="video_with_skeleton.mp4", out_plot="keypoints_animation.mp4", num_joints=17, fps=25):
    # === Caricamento video ===
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Ridimensiona per YOLO (640x640)
        frame_resized = cv2.resize(frame, (640, 480))
        frames.append(frame_resized)
    cap.release()

    frames = np.array(frames)
    frames_batch = np.expand_dims(frames, axis=0)  # (1, T, H, W, 3)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]

    # === Estrazione keypoints ===
    print(f"[INFO] Inizializzazione estrattore keypoints per frame {H}x{W}")
    extractor_seq = KeypointExtractorSeq(num_joints=num_joints, frame_size=(H, W))
    print("[INFO] Estrazione keypoints...")
    keypoints_seq = extractor_seq(frames_batch)[0]  # (T, J, 3)
    print(f"[DEBUG] Shape keypoints_seq: {keypoints_seq.shape}")

    # === Writer per video con skeleton ===
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_skel = cv2.VideoWriter(out_skeleton, fourcc, fps, (W, H))

    for t in range(T):
        frame = frames[t].copy()
        frame = draw_skeleton(frame, keypoints_seq[t], H, W)
        out_skel.write(frame)
    out_skel.release()
    print(f"[INFO] Video con skeleton salvato in {out_skeleton}")

    # === Writer per video cartesiano ===
    out_plotter = cv2.VideoWriter(out_plot, fourcc, fps, (640, 480))
    colors = plt.cm.get_cmap("tab20", num_joints)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.set_xlim(0,1)
        ax.set_ylim(1,0)
        ax.set_title(f"Frame {t}")
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Y (normalized)")
        ax.grid(True)

        for j in range(num_joints):
            x, y, conf = keypoints_seq[t, j]
            if conf > 0:
                ax.scatter(x, y, color=colors(j), s=40)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, (640, 480))
        out_plotter.write(img)
        plt.close(fig)

    out_plotter.release()
    print(f"[INFO] Video cartesiano salvato in {out_plot}")


if __name__ == "__main__":
    video_path = "AR17/code/video/IMG_4277.mp4"
    process_video(video_path)