import torch
import cv2
import numpy as np
from config import Config
from model import create_model
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sportspose.plot_skeleton_3d import plot_skeleton_3d
from learning import load_pretrained_weights

def joints3d_to_img(joints3d, size):
    """
    Crea immagine con scheletro 3D vero (non proiettato), con prospettiva,
    e ritorna np.uint8 (size, size, 3).
    """
    fig = plt.figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")

    plot_skeleton_3d(joints3d, ax=ax)

    # Imposta limiti uguali su x, y, z per evitare distorsioni
    max_range = (joints3d.max(axis=0) - joints3d.min(axis=0)).max() / 2.0
    mid = joints3d.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Abilita griglia e assi
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Scegli una vista “obliqua” / isometrica
    #ax.view_init(elev=20, azim=-20)

    # Forza proporzioni uniformi sugli assi
    ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()

    # Rendering in numpy
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = img[..., :3]
    plt.close(fig)

    # Ridimensiona a quadrato
    img = cv2.resize(img, (size, size))
    return img



# -------------------------------
# Inference su video
# -------------------------------
def run_inference(video_path, checkpoint_path, out_path="output.avi"):
    cfg = Config()

    model = create_model(inference=True)
    model = load_pretrained_weights(model, torch.load(checkpoint_path, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore apertura video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # === Lettura dei frame ===
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"[INFO] Video caricato: {len(frames)} frames di dimensione {width}x{height}")

    # === Inference ===
    with torch.no_grad():
        # 🔹 Importante: passiamo una lista di np.ndarray
        #   e NON np.array(frames), altrimenti YOLO crasha 
        frames_resized = [cv2.resize(f, (192, 256)) for f in frames]    # (T, H, W, 3)
        print("Frames resized shape:", np.array(frames_resized).shape)
        frames_batch = np.expand_dims(np.array(frames_resized), axis=0) # (1, T, H, W, 3)
        pred = model(frames_batch)  # (1, T, J, 3)

    preds_3d = pred[0].cpu().numpy()  # (T, J, 3)

    # === Scrittura del video con la posa 3D ===
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width * 2, height))

    for i, frame in enumerate(frames):
        joints = preds_3d[i]
        pose_img = joints3d_to_img(joints, size=frame.shape[0])
        pose_img_resized = cv2.resize(pose_img, (frame.shape[1], frame.shape[0]))

        # Se hai convertito a RGB, torna a BGR per OpenCV
        #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        combined = cv2.hconcat([frame, pose_img_resized])
        out.write(combined)

    out.release()
    print(f"[INFO] Video salvato in: {out_path}")



if __name__ == "__main__":
    weights_path = "weights/best_epoch.bin"  # Percorso ai pesi salvati
    video_path = "AR17/code/video/IMG_3990.mp4"  # Percorso al video di input
    run_inference(video_path, weights_path)
