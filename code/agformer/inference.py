import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from config import Config
from model import create_model
from learning import load_pretrained_weights


H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

def joints3d_to_img(joints3d, size=512, flip_y=True, flip_z=True):
    fig = plt.figure(figsize=(5, 5)) # Leggermente più grande per evitare tagli
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")

    points = joints3d.copy()
    if flip_y:
        points[:, 1] *= -1
    if flip_z:
        points[:, 2] *= -1

    # Disegnamo i joint
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, c='green', edgecolors='w')

    # Disegnamo le ossa
    for i, j in H36M_CONNECTIONS:
        ax.plot(points[[i, j], 0], points[[i, j], 1], points[[i, j], 2], color='red', linewidth=2)

    # --- CORREZIONE ORIENTAMENTO ---
    # Impostiamo Y come verticale e lo invertiamo per raddrizzare l'omino
    ax.view_init(elev=20, azim=-45, vertical_axis='y') 
    #ax.invert_yaxis() 

    # --- CORREZIONE TAGLIO (BOX SIZE) ---
    # Calcoliamo il centro e il range
    mid = points.mean(axis=0)
    # Aggiungiamo un 20% di margine (1.2) per evitare che i piedi o la testa tocchino i bordi
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 1.6 

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Estetica assi
    ax.set_axis_on()
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Altezza)')
    ax.set_zlabel('Z')
    
    # Rimuove lo sfondo grigio dei pannelli per chiarezza
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.tight_layout()
    canvas.draw()

    # Conversione in immagine OpenCV
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)

    return cv2.resize(img, (size, size))

# =========================
# Funzione principale
# =========================
def run_inference(video_path, checkpoint_path, out_path="output.avi", flip_z=True):
    cfg = Config()

    # Carica modello
    model = create_model((352, 640))
    #model = create_model(cfg.frame_size_sport)

    model = load_pretrained_weights(model, checkpoint_path, cfg.device)
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
    frames_resized = [cv2.resize(f, (640, 352)) for f in frames]
    #frames_resized = [cv2.resize(f, (416, 640)) for f in frames]
    frames_batch = np.array(frames_resized)
    frames_batch = torch.from_numpy(frames_batch).float().unsqueeze(0)
    print(f"[INFO] Batch preparato: {frames_batch.shape}")

    # Supponiamo frames_batch sia (1, T, 352, 640, 3) 
    # Dopo il preprocessing diventa (1, T, J, C) -> es. (1, 91, 17, 3)
    target_frames = 243
    current_frames = frames_batch.shape[1]

    if current_frames < target_frames:
        # Opzione A: Padding con l'ultimo frame (ripetizione)
        padding_size = target_frames - current_frames
        last_frame = frames_batch[:, -1:, :, :, :]
        padding = last_frame.repeat(1, padding_size, 1, 1, 1)
        frames_batch = torch.cat([frames_batch, padding], dim=1)
    elif current_frames > target_frames:
        # Opzione B: Sampling/Interpolazione
        # (Richiede di gestire i tensori con cura per non perdere dati)
        frames_batch = frames_batch[:, :target_frames, :, :, :]
    print(f"[INFO] Batch preparato: {frames_batch.shape}")

    # Inferenza
    with torch.no_grad():
        pred = model(frames_batch) * cfg.scale_factor  # (1, T, J, 3)
    preds_3d = pred[0].cpu().numpy()  # (T, J, 3)
    print(f"[INFO] Predizioni ottenute: {preds_3d.shape}")

    # Scrittura video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width * 2, height))

    num_frames = min(len(frames), preds_3d.shape[0])

    for i in range(num_frames):
        frame = frames[i]
        joints = preds_3d[i]
        pose_img = joints3d_to_img(joints, size=frame.shape[0], flip_z=flip_z)
        pose_img_resized = cv2.resize(pose_img, (frame.shape[1], frame.shape[0]))
        combined = cv2.hconcat([frame, pose_img_resized])
        out.write(combined)

        if i % 20 == 0:
            print(f"[INFO] Frame {i}/{num_frames} elaborato...")

    out.release()
    print(f"[INFO] Video salvato in: {out_path}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    weights_path = "experiments/motionagformer/motionagformer-b-h36m.pth.tr"  # modello preaddestrato
    #video_path = "AR17/code/video/IMG_3990.mp4"
    video_path = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/data/train_set/S1/Lutz_4_cam_1.mp4"
    run_inference(video_path, weights_path, out_path="inference.avi", flip_z=True)
