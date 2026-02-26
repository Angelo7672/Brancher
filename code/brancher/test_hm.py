import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

from config import Config
from model import create_model
from learning import *
from sportspose.plot_skeleton_3d import plot_skeleton_3d

def extract_keypoints_from_heatmaps(heatmaps, target_size):
    """
    Estrae le coordinate (x, y) tramite Argmax per ogni joint.
    heatmaps: [J, H_hm, W_hm]
    target_size: (width_img, height_img)
    """
    J, H_hm, W_hm = heatmaps.shape
    W_img, H_img = target_size
    
    keypoints = []
    
    for j in range(J):
        hm = heatmaps[j]
        # Trova l'indice del valore massimo
        idx = torch.argmax(hm.view(-1))
        y_hm = (idx // W_hm).item()
        x_hm = (idx % W_hm).item()
        
        # Mappa le coordinate sulla dimensione dell'immagine originale
        x_img = int((x_hm / (W_hm - 1)) * W_img)
        y_img = int((y_hm / (H_hm - 1)) * H_img)
        
        keypoints.append((x_img, y_img))
        
    return keypoints

def visualize_heatmaps(heatmaps, target_size):
    # heatmaps: (J_subset, H, W)
    
    # Invece della somma, prendiamo il valore massimo tra i joint per ogni pixel
    # Questo evita che le zone di sovrapposizione diventino macchie informi
    agg_hm, _ = torch.max(heatmaps, dim=0)
    agg_hm = agg_hm.numpy()
    
    # Normalizzazione robusta
    agg_hm = np.maximum(agg_hm, 0)
    if agg_hm.max() > 0:
        agg_hm = agg_hm / agg_hm.max()
    
    # Applichiamo una soglia: i valori molto bassi (rumore) li portiamo a zero
    agg_hm[agg_hm < 0.1] = 0
    
    agg_hm = (agg_hm * 255).astype(np.uint8)
    agg_hm_resized = cv2.resize(agg_hm, target_size)
    heatmap_color = cv2.applyColorMap(agg_hm_resized, cv2.COLORMAP_JET)
    
    # Opzionale: rendiamo nero lo sfondo dove la heatmap è zero
    heatmap_color[agg_hm_resized == 0] = 0
    
    return heatmap_color

# Modifica nella funzione run_inference
def run_inference(video_path, checkpoint_path, out_path="output.avi", flip_z=True):
    cfg = Config()
    model = create_model(frame_size = (640,352),inference=True)
    # Assicurati che il caricamento pesi gestisca la nuova UncertaintyHead
    model = load_weights(model, checkpoint_path, cfg.device)
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
    
    # Prepara batch per inferenza
    frames_resized = [cv2.resize(f, (352, 640)) for f in frames]
    frames_batch = np.array(frames_resized)
    frames_batch = torch.from_numpy(frames_batch).float().unsqueeze(0)
    print(f"[INFO] Batch preparato: {frames_batch.shape}")

    # Supponiamo frames_batch sia (1, T, 352, 640, 3) 
    # Dopo il preprocessing diventa (1, T, J, C) -> es. (1, 91, 17, 3)
    target_frames = 81
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

    with torch.no_grad():
        _, _, pred_hm, pred_log_var = model(frames_batch) 
    
    preds_hm = pred_hm[0].cpu() 
    
    # --- NUOVA LOGICA: SALVATAGGIO 17 HEATMAPS SUL DECIMO FRAME ---
    idx_frame = 9 # Decimo frame
    if preds_hm.shape[0] > idx_frame:
        print(f"[INFO] Generazione 17 heatmap per il frame {idx_frame}...")
        
        # Creiamo una sottocartella per non intasare la root
        os.makedirs("joint_heatmaps", exist_ok=True)
        
        # Prendiamo il frame originale (decimo)
        original_frame_10 = frames[idx_frame]
        h_orig, w_orig = original_frame_10.shape[:2]
        
        # Iteriamo su ogni joint (17)
        for j in range(preds_hm.shape[1]): # J = 17
            # Estraiamo la singola heatmap per il joint j
            single_hm = preds_hm[idx_frame, j].numpy()
            
            # Normalizzazione della singola heatmap
            single_hm = np.maximum(single_hm, 0)
            if single_hm.max() > 0:
                single_hm = single_hm / single_hm.max()
            
            # Colorazione
            single_hm_uint8 = (single_hm * 255).astype(np.uint8)
            single_hm_resized = cv2.resize(single_hm_uint8, (w_orig, h_orig))
            heatmap_color = cv2.applyColorMap(single_hm_resized, cv2.COLORMAP_JET)
            
            # Rendi nero lo sfondo dove la confidenza è bassa (opzionale)
            heatmap_color[single_hm_resized < 30] = 0
            
            # Overlay con il frame originale
            overlay = cv2.addWeighted(original_frame_10, 0.6, heatmap_color, 0.4, 0)
            
            # Salvataggio
            file_name = f"joint_heatmaps/joint_{j:02d}_overlay.png"
            cv2.imwrite(file_name, overlay)
            
        print(f"[INFO] Completato! 17 immagini salvate nella cartella 'joint_heatmaps/'")
    else:
        print("[ERRORE] Il video è troppo corto per estrarre il decimo frame.")

    # --- SCRITTURA VIDEO ---
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    combined_width = width * 2
    out = cv2.VideoWriter(out_path, fourcc, fps, (combined_width, height))

    # Assicuriamoci di non iterare oltre i frame di heatmap prodotti dal modello
    num_output_frames = min(len(frames), preds_hm.shape[0])
    print(f"[INFO] Scrivendo video con {num_output_frames} frame (limitato da heatmap prodotte)")
    for i in range(num_output_frames):
        # Per il video usiamo tutti i joint o solo quelli del corpo?
        # Usiamo preds_hm[i, 5:] per coerenza con quello che la rete "sente" nella loss
        hm_img = visualize_heatmaps(preds_hm[i, :], (width, height))
        hm_overlay = cv2.addWeighted(frames[i], 0.6, hm_img, 0.4, 0)

        combined = cv2.hconcat([frames[i], hm_overlay])
        out.write(combined)

    out.release()
    print(f"[INFO] Video salvato in: {out_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    weights_path = "experiments/brancher/rotationbranch/best_weight_self.pth"  # modello preaddestrato
    video_path = "AR17/code/video/IMG_0005.mp4"
    run_inference(video_path, weights_path, out_path="output_sportspose.avi", flip_z=True)
