import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Import dai tuoi file/moduli
from config import Config
from model import create_model
from learning import load_weights
from keypoint_extractor import KeypointExtractorSeq

# --- AGGIUNTA: FUNZIONE PER DISEGNARE LO SCHELETRO 2D ---
def draw_skeleton_2d(frame, joints_2d, connections, color=(0, 255, 0), thickness=2):
    """
    Disegna lo scheletro 2D sopra un frame OpenCV.
    joints_2d: (17, 2) o (17, 3) dove le prime due sono x, y
    """
    for i, j in connections:
        # Prendiamo le coordinate (x, y) per i due giunti
        pt1 = tuple(joints_2d[i][:2].astype(int))
        pt2 = tuple(joints_2d[j][:2].astype(int))
        
        # Disegnamo il segmento se i punti sono validi (es. non negativi)
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
            
    # Disegnamo i nodi (joint)
    for pt in joints_2d:
        center = tuple(pt[:2].astype(int))
        if center[0] > 0 and center[1] > 0:
            cv2.circle(frame, center, thickness + 1, (0, 0, 255), -1)

# --- FUNZIONE 3D ORIGINALE ---
H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

def joints3d_to_img(joints3d, size, flip_y=True):
    fig = plt.figure(figsize=(7.5, 7.5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")

    points = joints3d.copy()
    if flip_y: points[:, 1] *= -1

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100, c='red', edgecolors='w')
    for i, j in H36M_CONNECTIONS:
        ax.plot(points[[i, j], 0], points[[i, j], 1], points[[i, j], 2], color='lime', linewidth=4)

    ax.view_init(elev=10, azim=-45, vertical_axis='y') 
    mid = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0 
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_axis_on()

    plt.tight_layout()
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return cv2.resize(img, size)

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    if torch.is_tensor(data): flipped_data = data.clone()
    else: flipped_data = data.copy()
    flipped_data[..., 0] *= -1  
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data

# =========================
# Funzione principale Modificata
# =========================
def run_inference(video_path, checkpoint_path, out_path="inference.mp4"):
    cfg = Config()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_resolution = 640 
    scale = target_resolution / max(width, height)
    new_width = int(round((width * scale) / 32) * 32)
    new_height = int(round((height * scale) / 32) * 32)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    frames_resized = [cv2.resize(f, (new_width, new_height)) for f in frames]
    frames_batch = np.array(frames_resized)
    frames_batch = torch.from_numpy(frames_batch).float().unsqueeze(0) # (1, T, H, W, 3)

    target_frames = 81
    current_frames = frames_batch.shape[1]
    
    # Gestione Stride/Padding
    if current_frames < target_frames:
        padding_size = target_frames - current_frames
        padding = frames_batch[:, -1:].repeat(1, padding_size, 1, 1, 1)
        frames_batch = torch.cat([frames_batch, padding], dim=1)
        applied_stride = 1
    else:
        applied_stride = max(1, current_frames // target_frames)
        frames_batch = frames_batch[:, :target_frames * applied_stride:applied_stride, :, :, :]

    # --- ESTRAZIONE POSA 2D (Nuova Logica) ---
    print("[INFO] Estrazione pose 2D...")
    # Inizializziamo l'estrattore 2D (basato sulle tue classi)
    # Assumiamo che KeypointExtractorSeq sia nel tuo scope
    extractor_2d = KeypointExtractorSeq(frame_size=(new_height, new_width), device=cfg.device)
    
    # Per ottenere coordinate pixel, sovrascriviamo temporaneamente il normalizer o 
    # usiamo una versione modificata del call. Qui forziamo l'uso di coordinate raw:
    with torch.no_grad():
        # Passiamo il batch di frame (già ridimensionati) all'estrattore 2D
        # extractor_2d.__call__ si aspetta (B, T, H, W, 3)
        preds_2d_tensor = extractor_2d(frames_batch) 
        # Importante: se il tuo normalizer nel codice originale scala in [-1, 1], 
        # dovresti modificare la classe KeypointExtractorSeq per restituire i pixel.
        # Se non l'hai modificata, convertiamo i normalizzati indietro in pixel:
        preds_2d = preds_2d_tensor[0].cpu().numpy()
        
        # Se sono normalizzati (tra -1 e 1), riportiamoli a coordinate pixel del frame ridimensionato:
        # Nota: Questo passaggio serve solo se il tuo extractor applica KeypointNormalizer
        if preds_2d.min() < 0: 
            preds_2d[..., 0] = (preds_2d[..., 0] + 1) * new_width / 2
            preds_2d[..., 1] = (preds_2d[..., 1] + (new_height/new_width)) * new_width / 2

    # --- INFERENZA 3D ---
    model_3d = create_model(frame_size=(new_height, new_width), inference=True)
    model_3d = load_weights(model_3d, checkpoint_path, cfg.device)
    model_3d.to(cfg.device).eval()

    with torch.no_grad():
        pred_orig, _, _, _ = model_3d(frames_batch)
        frames_flipped = torch.flip(frames_batch, dims=[3]) 
        pred_flipped_raw, _, _, _ = model_3d(frames_flipped)
        pred_flipped_corrected = flip_data(pred_flipped_raw)
        preds_3d_tensor = (pred_orig + pred_flipped_corrected) / 2
        
    preds_3d = preds_3d_tensor[0].cpu().numpy()

    # --- SCRITTURA VIDEO ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_width * 2, new_height))

    for i in range(target_frames):
        idx_originale = min(i * applied_stride, len(frames_resized) - 1)
        
        # Frame originale ridimensionato
        frame_to_show = frames_resized[idx_originale].copy()
        
        # 1. DISEGNA SCHELETRO 2D SUL FRAME ORIGINALE
        current_2d = preds_2d[i] # Già mappato su H36M dall'extractor
        draw_skeleton_2d(frame_to_show, current_2d, H36M_CONNECTIONS)
        
        # 2. GENERA IMMAGINE 3D
        joints_3d = preds_3d[i]
        pose_img = joints3d_to_img(joints_3d, size=(new_width, new_height))
        
        # Concatenazione
        combined = cv2.hconcat([frame_to_show, pose_img])
        out.write(combined)

    out.release()
    print(f"[INFO] Video salvato: {out_path}")

if __name__ == "__main__":
    weights_path = "experiments/brancher/model_20260225-003614/best_weight.pth"  # modello preaddestrato
    video_path = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/data/train_set/S3/Running_13_cam_2.mp4"
    run_inference(video_path, weights_path, out_path="output_2d_3d.mp4")