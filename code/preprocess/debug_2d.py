import json
import cv2
import torch
from torchvision.io import read_video

# ========== CONFIG ==========
video_path = "/Experiments/angeloa00/AirPose/train/block_000/video.mp4"
pose2d_path = "/Experiments/angeloa00/AirPose_COCO/train/block_000/keypoints2d.json"
output_path = "video_with_keypoints.mp4"

# Cambia colore dei punti (B, G, R)
COLOR = (0, 255, 0)  # verde
RADIUS = 3
THICKNESS = -1
# ============================


# === Carica video ===
video_frames, _, info = read_video(video_path, output_format="THWC", pts_unit="sec")
video_frames = video_frames.to(torch.uint8)
T_video = video_frames.shape[0]

print("Video shape:", video_frames.shape)
fps = int(info["video_fps"])
H, W = video_frames.shape[1:3]

# === Carica keypoints ===
with open(pose2d_path, "r") as f:
    pose2d = json.load(f)

keypoints_all = pose2d#pose2d["keypoints"]   # lista lunga T, ogni elemento: J×3

T_kp = len(keypoints_all)
J = len(keypoints_all[0])

print(f"Loaded {T_kp} frames of 2D keypoints, {J} joints each.")

if T_kp != T_video:
    print("⚠ WARNING: Frame mismatch! Video frames != Keypoint frames")


# === Setup writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))


# === Funzione per disegnare keypoints ===
def draw_keypoints(frame_bgr, keypoints):
    """
    frame_bgr: numpy [H,W,3] BGR
    keypoints: lista di [x,y,conf]
    """
    for (x, y, conf) in keypoints:
        if conf < 1e-6:   # punto non trovato
            continue
        x = int(x)
        y = int(y)
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(frame_bgr, (x, y), RADIUS, COLOR, THICKNESS)
    return frame_bgr


# === Generazione video ===
for t in range(min(T_video, T_kp)):
    frame = video_frames[t].numpy()     # THWC → HWC
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    kpts = keypoints_all[t]  # lista J×3

    frame_bgr = draw_keypoints(frame_bgr, kpts)

    out.write(frame_bgr)

out.release()
print(f"Video con keypoints salvato in: {output_path}")
