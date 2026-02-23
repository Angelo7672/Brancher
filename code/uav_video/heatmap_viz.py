import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def project_heatmap_to_full_frame(local_hms, bbox, frame_size):
    H_full, W_full = frame_size
    num_joints = local_hms.shape[0]
    
    global_canvas = np.zeros((num_joints, H_full, W_full), dtype=np.float32)
    
    # Arrotondiamo e convertiamo in int
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    bw = x2 - x1
    bh = y2 - y1

    # OpenCV ha bisogno di float32, assicuriamoci che lo sia
    local_hms = local_hms.astype(np.float32)

    if bw > 0 and bh > 0:
        for j in range(num_joints):
            local_hm = local_hms[j]
            
            # cv2.resize si aspetta (width, height)
            try:
                resized_hm = cv2.resize(local_hm, (bw, bh), interpolation=cv2.INTER_LINEAR)
                
                # Clipping per sicurezza rispetto ai bordi del frame
                y_start = max(0, y1)
                y_end = min(H_full, y1 + bh)
                x_start = max(0, x1)
                x_end = min(W_full, x1 + bw)
                
                # Calcoliamo quanto della heatmap ridimensionata usare 
                # (nel caso il bbox esca dai bordi del video)
                hm_y_end = y_end - y_start
                hm_x_end = x_end - x_start
                
                if hm_y_end > 0 and hm_x_end > 0:
                    global_canvas[j, y_start:y_end, x_start:x_end] = resized_hm[:hm_y_end, :hm_x_end]
            except Exception as e:
                print(f"Errore nel resize del giunto {j}: {e}")
                continue
            
    return global_canvas

def visualize_full_frame_projection(sample_dir, video_path, output_path="full_frame_projected.mp4", fps=30):
    # --- 1. Caricamento Video ---
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.array(frames)
    H_vid, W_vid = video.shape[1], video.shape[2]

    # --- 2. Caricamento Dati ---
    with open(os.path.join(sample_dir, "bboxes.json"), "r") as f:
        bboxes = np.array(json.load(f))
    
    heatmaps_tensor = torch.load(os.path.join(sample_dir, "heatmaps.pt"), map_location='cpu')
    heatmaps_raw = heatmaps_tensor.numpy() if isinstance(heatmaps_tensor, torch.Tensor) else heatmaps_tensor

    T = min(len(video), len(heatmaps_raw), len(bboxes))
    
    # --- 3. Plotting ---
    writer = FFMpegWriter(fps=fps)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    print(f"Generazione video proiettato... Risoluzione: {W_vid}x{H_vid}")
    
    with writer.saving(fig, output_path, dpi=100):
        for t in range(T):
            ax.clear()
            
            # Proiettiamo le heatmap locali sul frame intero
            # Questa è l'operazione che volevi automatizzare
            full_frame_hms = project_heatmap_to_full_frame(heatmaps_raw[t], bboxes[t], (H_vid, W_vid))
            
            # Creiamo una versione visualizzabile (max dei canali)
            combined_hm = np.max(full_frame_hms, axis=0)
            
            # Normalizzazione per il plot
            if combined_hm.max() > 0:
                combined_hm /= combined_hm.max()
            
            # Applichiamo maschera per trasparenza
            combined_hm = np.ma.masked_where(combined_hm < 0.2, combined_hm)

            # Mostriamo il video
            ax.imshow(video[t])
            
            # Mostriamo la heatmap proiettata (senza usare 'extent'!)
            # Ora la heatmap è già grande quanto il video
            ax.imshow(combined_hm, cmap='jet', alpha=0.5)
            
            # Disegno il BBox solo per conferma visiva
            x1, y1, x2, y2 = bboxes[t]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='white', linewidth=1, alpha=0.3)
            ax.add_patch(rect)

            ax.set_title(f"Frame {t} - Heatmap Proiettata su Frame Intero ({W_vid}x{H_vid})")
            ax.axis("off")
            writer.grab_frame()
            
    plt.close(fig)
    print(f"Visualizzazione completata in: {output_path}")

if __name__ == "__main__":
    SAMPLE_DIR = "/mnt/proj2/eu-25-52/ar17/DroneDataset/train/video_00715"
    VIDEO_SOURCE = "/mnt/proj2/eu-25-52/ar17/DroneDataset/video_and_flipped/video_0715.mp4"
    visualize_full_frame_projection(SAMPLE_DIR, VIDEO_SOURCE)