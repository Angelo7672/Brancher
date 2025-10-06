import cv2
import numpy as np
import matplotlib.pyplot as plt
from keypoint_extractor import KeypointExtractorSeq

# COCO keypoints names e skeleton

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def draw_skeleton(frame, keypoints, conf_thresh=0.3):
    """
    Disegna i keypoint e lo scheletro sul frame.
    """
    for j, (x, y, c) in enumerate(keypoints):
        if c > conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(frame, str(j), (int(x)+5, int(y)+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    for (j1, j2) in COCO_PAIRS:
        if keypoints[j1][2] > conf_thresh and keypoints[j2][2] > conf_thresh:
            p1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
            p2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
    return frame


def process_video(video_path, out_skeleton="video_with_skeleton.mp4", out_plot="keypoints_animation.mp4", num_joints=17, fps=25):
    # === Caricamento video ===
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frames = np.array(frames)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"[INFO] Video caricato: {T} frames di dimensione {W}x{H}")
    frames_batch = np.expand_dims(frames, axis=0)

    # === Estrazione keypoints ===
    extractor_seq = KeypointExtractorSeq(num_joints=num_joints, frame_size=(256, 192))
    print("Shape:", frames_batch.shape)
    print("Extracting keypoints...", extractor_seq(frames_batch).shape)
    keypoints_seq = extractor_seq(frames_batch)[0]  # (T, J, 3)

    print("\n[DEBUG] Shape keypoints_seq:", keypoints_seq.shape)
 
    


    # === Writer per video con skeleton ===
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_skel = cv2.VideoWriter(out_skeleton, fourcc, fps, (W, H))

    for t in range(T):
        frame = frames[t].copy()
        frame = draw_skeleton(frame, keypoints_seq[t])
        out_skel.write(frame)
    out_skel.release()
    print(f"[INFO] Video con skeleton salvato in {out_skeleton}")

    # === Writer per video cartesiano ===
    out_plotter = cv2.VideoWriter(out_plot, fourcc, fps, (640, 480))
    xs_all = keypoints_seq[..., 0]
    ys_all = keypoints_seq[..., 1]
    xs_min, xs_max = np.min(xs_all[xs_all > 0]), np.max(xs_all)
    ys_min, ys_max = np.min(ys_all[ys_all > 0]), np.max(ys_all)

    colors = plt.cm.get_cmap("tab20", num_joints)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.set_xlim(xs_min - 1, xs_max + 1)
        ax.set_ylim(ys_max + 1, ys_min - 1)
        ax.set_title(f"Frame {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        for j in range(num_joints):
            x, y, conf = keypoints_seq[t, j]
            if conf > 0:
                ax.scatter(x, y, color=colors(j), s=40)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 480))
        out_plotter.write(img)
        plt.close(fig)

    out_plotter.release()
    print(f"[INFO] Video cartesiano salvato in {out_plot}")


if __name__ == "__main__":
    video_path = "AR17/code/video/IMG_3990.mp4"
    process_video(video_path)



