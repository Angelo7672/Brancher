import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from config import Config
from model import create_model
from learning import load_pretrained_weights
from sportspose.plot_skeleton_3d import plot_skeleton_3d


# =========================
# Funzione: joints3d_to_img
# =========================
def joints3d_to_img(joints3d, size=512, flip_z=True):
    """
    Crea immagine RGB con lo scheletro 3D disegnato con la funzione di SportsPose.
    """
    fig = plt.figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")

    # Usa la funzione ufficiale di SportsPose
    plot_skeleton_3d(joints3d, ax=ax, flip_z=flip_z)

    # Imposta limiti uniformi sugli assi (evita distorsione)
    max_range = (joints3d.max(axis=0) - joints3d.min(axis=0)).max() / 2.0
    mid = joints3d.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Ruota la vista
    ax.view_init(elev=20, azim=0)

    ax.grid(False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()

    # Rendering → numpy
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = img[..., :3]
    plt.close(fig)

    img = cv2.resize(img, (size, size))
    return img


# =========================
# Funzione principale
# =========================
def run_inference(video_path, checkpoint_path, out_path="output.avi", flip_z=True):
    cfg = Config()

    # Carica modello
    model = create_model()
    model = load_pretrained_weights(model, torch.load(checkpoint_path, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    # Carica video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"[INFO] Video caricato: {len(frames)} frame ({width}x{height})")

    # Prepara batch per inferenza
    frames_resized = [cv2.resize(f, (640, 416)) for f in frames]
    frames_batch = np.array(frames_resized)
    frames_batch = torch.from_numpy(frames_batch).float().unsqueeze(0)

    # Inferenza
    with torch.no_grad():
        pred = model(frames_batch)  # (1, T, J, 3)
    preds_3d = pred[0].cpu().numpy()  # (T, J, 3)
    print(f"[INFO] Predizioni ottenute: {preds_3d.shape}")

    # Scrittura video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width * 2, height))

    for i, frame in enumerate(frames):
        joints = preds_3d[i]
        pose_img = joints3d_to_img(joints, size=frame.shape[0], flip_z=flip_z)
        pose_img_resized = cv2.resize(pose_img, (frame.shape[1], frame.shape[0]))
        combined = cv2.hconcat([frame, pose_img_resized])
        out.write(combined)

        if i % 20 == 0:
            print(f"[INFO] Frame {i}/{len(frames)} elaborato...")

    out.release()
    print(f"[INFO] Video salvato in: {out_path}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    weights_path = "/Experiments/angeloa00/baseline_2d_lift/model_20251111-183337/best_weight.pth"  # modello preaddestrato
    #video_path = "AR17/code/video/IMG_3990.mp4"
    video_path = "/DATA/SportsPose/videos/indoors/S00/Video_14_10_2022_103104/CAM0.avi"
    run_inference(video_path, weights_path, out_path="output_sportspose.avi", flip_z=True)
