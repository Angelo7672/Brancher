import os
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- CONFIGURAZIONE PERCORSI ---
PKL_PATH = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/frame_81/train/00000002.pkl' 
PKL_PATH2 = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/pose_3d_v3/frame_81/train/00000002.pkl' 
HMAP_ROOT = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/heatmap'
OUTPUT_NAME = 'debug_3d_matplotlib.mp4'
FPS = 20

def get_h36m_edges():
    return [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), (8,9), (9,10), 
            (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]

def render_3d_panel(joints3d, edges, size=(640, 480)):
    """
    Crea il pannello 3D usando Matplotlib con assi orientati correttamente.
    """
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')

    points = joints3d.copy()
    
    # --- CORREZIONE ORIENTAMENTO ---
    # Invertiamo Y per raddrizzare l'atleta (standard OpenCV -> Matplotlib Up)
    points[:, 1] *= -1 

    # Disegno Joint
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=20)

    # Disegno Ossa
    for start, end in edges:
        ax.plot(points[[start, end], 0], points[[start, end], 1], points[[start, end], 2], color='blue', linewidth=2)

    # --- IMPOSTAZIONI ASSI ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Altezza)')
    ax.set_zlabel('Z')
    
    # Manteniamo le proporzioni degli assi uguali (evita distorsioni)
    mid = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 1.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Vista diagonale per vedere bene la profondità
    ax.view_init(elev=15, azim=-45, vertical_axis='y')

    plt.tight_layout()
    canvas.draw()
    
    # Conversione in immagine OpenCV
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(size[1], size[0], 4)
    return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)

def draw_skeleton_2d(canvas, joints, edges, color=(0, 255, 0), label=""):
    for start, end in edges:
        p1 = (int(joints[start, 0]), int(joints[start, 1]))
        p2 = (int(joints[end, 0]), int(joints[end, 1]))
        cv2.line(canvas, p1, p2, color, 2)
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    # 1. Caricamento Dati
    if not os.path.exists(PKL_PATH):
        print(f"❌ File non trovato: {PKL_PATH}")
        return

    with open(PKL_PATH, 'rb') as f:
        data_dict = pickle.load(f)
    with open(PKL_PATH2, 'rb') as f:
        data_dict2 = pickle.load(f)
    
    inputs = data_dict2['data_input']   # [81, 17, 3]
    labels = data_dict2['data_label']   # [81, 17, 3]
    hmap_path = os.path.join(HMAP_ROOT, data_dict['hmap_relative_path'], data_dict['hmap_file'])
    all_heatmaps = torch.load(hmap_path, map_location='cpu')
    hmap_indices = data_dict['hmap_indices']

    # 2. Configurazione Video
    w_panel, h_panel = 640, 480
    out = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w_panel * 3, h_panel))
    edges = get_h36m_edges()

    print(f"🎬 Generazione video: {OUTPUT_NAME}")

    for i in range(inputs.shape[0]):
        # --- PANNELLO 1: INPUT 2D ---
        panel_2d = np.zeros((h_panel, w_panel, 3), dtype=np.uint8)
        pts_2d = (inputs[i, :, :2] + 1) / 2 * np.array([w_panel, h_panel])
        draw_skeleton_2d(panel_2d, pts_2d, edges, (0, 255, 0), "Input 2D")

        # --- PANNELLO 2: HEATMAP ---
        idx = hmap_indices[i]
        # 1. Converti esplicitamente in float32 (OpenCV amichevole)
        hmap_sum = torch.sum(all_heatmaps[idx], dim=0).detach().cpu().to(torch.float32).numpy()
        # 2. Ora la normalizzazione funzionerà senza errori
        hmap_norm = cv2.normalize(hmap_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        panel_hmap = cv2.applyColorMap(hmap_norm, cv2.COLORMAP_JET)
        panel_hmap = cv2.resize(panel_hmap, (w_panel, h_panel))
        cv2.putText(panel_hmap, f"Heatmap Index: {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- PANNELLO 3: RENDERING 3D (Matplotlib) ---
        # Usiamo 'labels' che contiene le coordinate 3D complete
        panel_3d = render_3d_panel(labels[i], edges, size=(w_panel, h_panel))

        # Unione e scrittura
        combined = np.hstack([panel_2d, panel_hmap, panel_3d])
        out.write(combined)
        
        if i % 10 == 0: print(f"Elaborazione frame {i}/{inputs.shape[0]}...")

    out.release()
    print(f"✅ Completato: {OUTPUT_NAME}")

if __name__ == "__main__":
    main()