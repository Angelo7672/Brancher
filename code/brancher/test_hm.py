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
    model = create_model(cfg.frame_size_sport)
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
    
    # Prepara batch per inferenza
    frames_resized = [cv2.resize(f, (416, 640)) for f in frames]
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

    with torch.no_grad():
        # Ora il modello restituisce due output
        _,_, pred_hm, pred_log_var = model(frames_batch) 
        
    # Prendiamo il primo video del batch e portiamolo su CPU
    preds_hm = pred_hm[0].cpu() 
    preds_log_var = pred_log_var[0].cpu() # [T, J]


    # --- DEBUG MATEMATICO: Salvataggio Raw ---
    # Salviamo il primo frame (T=0)
    # Salviamo TUTTI i joint (anche i primi 5) per capire come si comporta la faccia
    torch.save(preds_hm[0], "debug_hm_ours_raw.pt")
    torch.save(preds_log_var[0], "debug_logvar_ours_raw.pt")
    
    print(f"[DEBUG] Salvato tensore heatmap {preds_hm[0].shape}")
    print(f"[DEBUG] Salvato tensore incertezza {preds_log_var[0].shape}")




    # --- SALVATAGGIO PRIMO FRAME PER CONFRONTO ---
    first_frame = frames[0]
    # Visualizziamo solo i joint del corpo (indice 5+) come deciso per la loss
    # Se vuoi vederli tutti, usa: preds_hm[0]
    hm_first_frame = visualize_heatmaps(preds_hm[0, 5:], (width, height))
    
    # Overlay del primo frame
    overlay_first = cv2.addWeighted(first_frame, 0.5, hm_first_frame, 0.5, 0)
    
    # Salva le immagini singole per il confronto
    cv2.imwrite("comparison_ours_raw.png", first_frame)
    cv2.imwrite("comparison_ours_heatmap.png", hm_first_frame)
    cv2.imwrite("comparison_ours_overlay.png", overlay_first)
    print(f"[INFO] Frame di confronto salvati come 'comparison_ours_*.png'")

    # --- SCRITTURA VIDEO ---
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    combined_width = width * 2
    out = cv2.VideoWriter(out_path, fourcc, fps, (combined_width, height))

    for i in range(len(frames)):
        # Per il video usiamo tutti i joint o solo quelli del corpo? 
        # Usiamo preds_hm[i, 5:] per coerenza con quello che la rete "sente" nella loss
        hm_img = visualize_heatmaps(preds_hm[i, 5:], (width, height))
        hm_overlay = cv2.addWeighted(frames[i], 0.6, hm_img, 0.4, 0)
        
        combined = cv2.hconcat([frames[i], hm_overlay])
        out.write(combined)

    out.release()
    print(f"[INFO] Video salvato in: {out_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    weights_path = "experiments/brancher/model_20260206-234555/best_weight.pth"  # modello preaddestrato
    video_path = "AR17/code/video/IMG_0005.mp4"
    run_inference(video_path, weights_path, out_path="output_sportspose.avi", flip_z=True)
