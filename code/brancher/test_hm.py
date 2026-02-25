import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
        # Ora il modello restituisce due output
        _,_, pred_hm, pred_log_var = model(frames_batch) 
        
    # Prendiamo il primo video del batch e portiamolo su CPU
    preds_hm = pred_hm[0].cpu() 
    preds_log_var = pred_log_var[0].cpu() # [T, J]


    # --- DEBUG MATEMATICO: Salvataggio Raw ---
    # Salviamo il primo frame (T=0)
    # Salviamo TUTTI i joint (anche i primi 5) per capire come si comporta la faccia
    #torch.save(preds_hm[0], "debug_hm_ours_raw.pt")
    #torch.save(preds_log_var[0], "debug_logvar_ours_raw.pt")
    
    print(f"[DEBUG] Salvato tensore heatmap {preds_hm[0].shape}")
    print(f"[DEBUG] Salvato tensore incertezza {preds_log_var[0].shape}")

    # --- NUOVA LOGICA: ARGMAX PER IL PRIMO FRAME ---
    first_frame_hms = preds_hm[0] # [17, H, W]
    kpts_first_frame = extract_keypoints_from_heatmaps(first_frame_hms, (width, height))
    
    print(f"[DEBUG] Estratti {len(kpts_first_frame)} keypoints tramite Argmax")


    # --- SALVATAGGIO PRIMO FRAME PER CONFRONTO ---
    first_frame = frames[0]
    # Visualizziamo solo i joint del corpo (indice 5+) come deciso per la loss
    # Se vuoi vederli tutti, usa: preds_hm[0]
    hm_first_frame = visualize_heatmaps(preds_hm[0, :], (width, height))
    
    # Overlay del primo frame
    overlay_first = cv2.addWeighted(first_frame, 0.5, hm_first_frame, 0.5, 0)
    
    # Salva le immagini singole per il confronto
    cv2.imwrite("comparison_ours_heatmap.png", hm_first_frame)
    cv2.imwrite("comparison_ours_overlay.png", overlay_first)
    print(f"[INFO] Frame di confronto salvati come 'comparison_ours_*.png'")

    # DISEGNO DEI PUNTI ARGMAX SULL'OVERLAY
    for i, (px, py) in enumerate(kpts_first_frame):
        # Disegna un cerchietto pieno per il giunto
        cv2.circle(overlay_first, (px, py), 4, (0, 255, 0), -1) 
        # Opzionale: aggiunge il numero del giunto per debug
        cv2.putText(overlay_first, str(i), (px+5, py-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Salva le immagini
    cv2.imwrite("comparison_ours_overlay_with_kpts.png", overlay_first)
    print(f"[INFO] Frame di confronto con keypoints salvato!")

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
    weights_path = "experiments/brancher/model_20260224-173154/weight_epoch41.pth"  # modello preaddestrato
    video_path = "AR17/code/video/IMG_0005.mp4"
    run_inference(video_path, weights_path, out_path="output_sportspose.avi", flip_z=True)
