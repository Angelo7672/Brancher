import cv2
import numpy as np
import matplotlib.pyplot as plt
from keypoint_extractor import KeypointExtractorSeq

# COCO keypoints names e skeleton
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def keypoints_to_video(video_path, output_path="keypoints_animation.mp4", num_joints=17, fps=25):
    # Carica video e frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frames = np.array(frames)
    B, T, H, W = 1, frames.shape[0], frames.shape[1], frames.shape[2]
    frames_batch = np.expand_dims(frames, axis=0)

    # Estrai keypoints
    extractor_seq = KeypointExtractorSeq(num_joints=num_joints, frame_size=(256, 192))
    keypoints_seq = extractor_seq(frames_batch)[0]  # (T, J, 3)

    # Prepara video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    # Calcola limiti fissi del plot
    xs_all = keypoints_seq[..., 0]
    ys_all = keypoints_seq[..., 1]
    xs_min, xs_max = np.min(xs_all[xs_all > 0]), np.max(xs_all)
    ys_min, ys_max = np.min(ys_all[ys_all > 0]), np.max(ys_all)

    colors = plt.cm.get_cmap("tab20", num_joints)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.set_xlim(xs_min-1, xs_max+1)
        ax.set_ylim(ys_max+1, ys_min-1)  # inverti y
        ax.set_title(f"Frame {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        # disegna keypoints
        for j in range(num_joints):
            x, y, conf = keypoints_seq[t, j]
            if conf > 0:
                ax.scatter(x, y, color=colors(j), s=40, label=COCO_NAMES[j] if t==0 else "")

        # salva figura come immagine RGB
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 480))

        out.write(img)
        plt.close(fig)

    out.release()
    print(f"[INFO] Video animato salvato in {output_path}")


if __name__ == "__main__":
    video_path = "AR17/code/video/IMG_3990.mp4"
    keypoints_to_video(video_path, output_path="keypoints_animation.mp4")


