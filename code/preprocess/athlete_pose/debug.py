import os
import pickle
import torch
import numpy as np
import cv2

# --- CONFIGURAZIONE PERCORSI ---
# Punta al nuovo .pkl generato con hmap_indices
PKL_PATH = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/frame_81/train/00006336.pkl' 
HMAP_ROOT = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/heatmap'
OUTPUT_NAME = 'debug_athlete_pose_new.mp4'
FPS = 20

def get_h36m_edges():
    return [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), (8,9), (9,10), 
            (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]

def draw_skeleton(canvas, joints, edges, color=(0, 255, 0), label=""):
    # joints: [17, 2] in range [0, view_size]
    for start, end in edges:
        p1 = (int(joints[start, 0]), int(joints[start, 1]))
        p2 = (int(joints[end, 0]), int(joints[end, 1]))
        cv2.line(canvas, p1, p2, color, 2)
    for j in range(joints.shape[0]):
        cv2.circle(canvas, (int(joints[j, 0]), int(joints[j, 1])), 3, (0, 0, 255), -1)
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main():
    # 1. Caricamento Metadata dal PKL
    if not os.path.exists(PKL_PATH):
        print(f"❌ File PKL non trovato: {PKL_PATH}")
        return

    with open(PKL_PATH, 'rb') as f:
        data_dict = pickle.load(f)
    
    inputs = data_dict['data_input']      # [81, 17, 3]
    labels = data_dict['data_label']      # [81, 17, 3]
    hmap_filename = data_dict['hmap_file'] 
    hmap_subfolder = data_dict['hmap_relative_path']
    hmap_indices = data_dict['hmap_indices'] # Gli indici corretti nel file .pt

    print(f"📄 Analisi clip: {os.path.basename(PKL_PATH)}")
    print(f"🔗 Puntatore Heatmap: {hmap_subfolder}/{hmap_filename}")

    # 2. Caricamento Heatmap Completa
    hmap_path = os.path.join(HMAP_ROOT, hmap_subfolder, hmap_filename)
    if not os.path.exists(hmap_path):
        print(f"❌ Errore: File Heatmap non trovato in {hmap_path}")
        return

    # Carichiamo il tensore gigante
    all_heatmaps = torch.load(hmap_path, map_location='cpu')
    
    # 3. Preparazione Video
    w_view, h_view = 640, 480
    out = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w_view * 3, h_view))
    edges = get_h36m_edges()

    T = inputs.shape[0] # Dovrebbe essere 81

    for i in range(T):
        # --- PANNELLO SINISTRA: 2D INPUT (Normalizzato) ---
        # I dati sono in [-1, 1], li portiamo in [0, w_view/h_view]
        # Formula: $pos\_pixel = \frac{pos\_norm + 1}{2} \cdot size$
        pts_2d = inputs[i, :, :2]
        pts_2d_vis = (pts_2d + 1) / 2 * np.array([w_view, h_view])
        
        panel_2d = np.zeros((h_view, w_view, 3), dtype=np.uint8)
        draw_skeleton(panel_2d, pts_2d_vis, edges, (0, 255, 0), "Input 2D (Normalized)")

        # --- PANNELLO CENTRO: HEATMAP CORRETTA ---
        idx_in_pt = hmap_indices[i]
        hmap = all_heatmaps[idx_in_pt] # [17, 64, 64]
        
        # Sommiamo i joint e convertiamo in numpy
        hmap_sum = torch.sum(hmap, dim=0).detach().cpu().numpy()
        
        # Controllo di sicurezza: se la heatmap è vuota o ha valori strani
        if np.max(hmap_sum) <= np.min(hmap_sum):
            # Crea un pannello nero di fallback se non c'è segnale
            panel_hmap = np.zeros((h_view, w_view, 3), dtype=np.uint8)
        else:
            # Normalizzazione sicura
            hmap_sum = ((hmap_sum - hmap_sum.min()) / (hmap_sum.max() - hmap_sum.min()) * 255).astype(np.uint8)
            panel_hmap = cv2.applyColorMap(hmap_sum, cv2.COLORMAP_JET)
            panel_hmap = cv2.resize(panel_hmap, (w_view, h_view))
            
        cv2.putText(panel_hmap, f"Hmap Index: {idx_in_pt}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- PANNELLO DESTRA: 3D LABEL PROJECTION ---
        pts_3d_proj = labels[i, :, :2]
        pts_3d_vis = (pts_3d_proj + 1) / 2 * np.array([w_view, h_view])
        
        panel_3d = np.zeros((h_view, w_view, 3), dtype=np.uint8)
        draw_skeleton(panel_3d, pts_3d_vis, edges, (255, 0, 0), "Label 3D (Projected)")

        # Combine
        combined = np.hstack([panel_2d, panel_hmap, panel_3d])
        out.write(combined)

    out.release()
    print(f"✅ Video di debug generato: {OUTPUT_NAME}")

if __name__ == "__main__":
    main()