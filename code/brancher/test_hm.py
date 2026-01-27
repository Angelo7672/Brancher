import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from config import Config
from model import create_model
from learning import load_pretrained_weights
from sportspose.plot_skeleton_3d import plot_skeleton_3d

def visualize_heatmaps(heatmaps, target_size):
    """
    heatmaps: tensor (J, H, W)
    target_size: (width, height) del frame originale
    """
    # 1. Somma tutte le heatmap dei giunti (J) per averne una globale
    # heatmaps è (J, H, W)
    agg_hm = torch.sum(heatmaps, dim=0).cpu().numpy()
    
    # 2. Normalizzazione per visualizzazione (0-255)
    agg_hm = np.maximum(agg_hm, 0) # Rimuovi eventuali valori negativi
    if agg_hm.max() > 0:
        agg_hm = agg_hm / agg_hm.max()
    agg_hm = (agg_hm * 255).astype(np.uint8)
    
    # 3. Resize alla dimensione del video
    agg_hm_resized = cv2.resize(agg_hm, target_size)
    
    # 4. Applica ColorMap (Jet è la classica blu-rosso)
    heatmap_color = cv2.applyColorMap(agg_hm_resized, cv2.COLORMAP_JET)
    
    return heatmap_color

# Modifica nella funzione run_inference
def run_inference(video_path, checkpoint_path, out_path="output.avi", flip_z=True):
    cfg = Config()
    model = create_model()
    # Assicurati che il caricamento pesi gestisca la nuova UncertaintyHead
    model = load_pretrained_weights(model, torch.load(checkpoint_path, map_location=cfg.device))
    model = model.to(cfg.device).eval()

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
    
    # Prepara batch
    frames_resized = [cv2.resize(f, (640, 416)) for f in frames]
    frames_batch = np.array(frames_resized)
    input_tensor = torch.from_numpy(frames_batch).float().unsqueeze(0)

    with torch.no_grad():
        # Ora il modello restituisce due output
        pred_3d, pred_hm = model(input_tensor) 
        
    preds_3d = (pred_3d[0] * cfg.scale_factor).cpu().numpy()
    preds_hm = pred_hm[0] # (T, J, H, W)

    # Scrittura video: facciamo 2 pannelli (Frame Originale | Heatmap)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    combined_width = width * 2
    out = cv2.VideoWriter(out_path, fourcc, fps, (combined_width, height))

    for i in range(len(frames)):
        frame = frames[i]
        
        # 1. Pannello Heatmap
        hm_img = visualize_heatmaps(preds_hm[i], (width, height))
        # Sovrapponiamo leggermente l'heatmap al frame originale per capire se i giunti "matchano"
        hm_overlay = cv2.addWeighted(frame, 0.6, hm_img, 0.4, 0)
        
        # Unisci tutto
        combined = cv2.hconcat([frame, hm_overlay])
        out.write(combined)

    out.release()

# =========================
# Main
# =========================
if __name__ == "__main__":
    weights_path = "experiments/brancher/model_20260114-164646/best_weight.pth"  # modello preaddestrato
    video_path = "AR17/code/video/IMG_0005.mp4"
    run_inference(video_path, weights_path, out_path="output_sportspose.avi", flip_z=True)
