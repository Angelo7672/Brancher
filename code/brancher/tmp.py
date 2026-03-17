import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keypoint_extractor import KeypointExtractorSeq

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === CONFIGURAZIONE PRESENTAZIONE (FULL HD) ===
MASTER_HEIGHT = 1080  # Altezza fissa per il video finale
GRAPH_WIDTH = 1080   # Il grafico sarà un quadrato 1080x1080
DPI = 150            # DPI alti per testi leggibili anche in fondo alla sala

input_video = "AR17/code/video/IMG_3990.mp4"
output_video = "output_ULTRA_LARGE_graph.mp4"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calcoliamo la nuova larghezza del video originale per mantenerlo proporzionato al MASTER_HEIGHT
video_display_w = int(orig_w * (MASTER_HEIGHT / orig_h))

# === Connessioni H36M ===
H36M_SKELETON = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), (8,9), (9,10), (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]

def render_massive_graph(kp, size):
    # Calcolo figsize dinamico
    f_size = size / DPI
    fig, ax = plt.subplots(figsize=(f_size, f_size), dpi=DPI)
    
    # Zoom che hai scelto
    ax.set_xlim(-1.1, 0.5)
    ax.set_ylim(-0.5, 0.25)
    ax.set_aspect('equal')
    
    # Label più grandi per le slide
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title('Normalized 2D Keypoints', fontsize=18, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Omino più spesso per essere visibile da lontano
    for i, j in H36M_SKELETON:
        if kp[i, 2] > 0.1 and kp[j, 2] > 0.1:
            ax.plot([kp[i, 0], kp[j, 0]], [-kp[i, 1], -kp[j, 1]], 
                    color='#E63946', linewidth=5, marker='o', 
                    markersize=8, markerfacecolor='#1D3557', markeredgecolor='white')

    plt.tight_layout()
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img

# --- Inferenza (Standard) ---
# ... (stessa parte del codice precedente per caricamento e extractor) ...
target_res = 640
scale = target_res / max(orig_w, orig_h)
new_w, new_h = int(round((orig_w * scale) / 32) * 32), int(round((orig_h * scale) / 32) * 32)
extractor = KeypointExtractorSeq(frame_size=(new_h, new_w), device=device)

frames_orig = []
frames_resized_inf = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames_orig.append(frame)
    frames_resized_inf.append(cv2.resize(frame, (new_w, new_h)))
cap.release()

video_tensor = torch.from_numpy(np.array(frames_resized_inf)).float().unsqueeze(0)
with torch.no_grad():
    keypoints = extractor(video_tensor)[0].cpu().numpy()

# === SCRITTURA VIDEO "CINEMATOGRAFICO" ===
# Larghezza totale = video scalato + grafico gigante
final_width = video_display_w + GRAPH_WIDTH
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (final_width, MASTER_HEIGHT))

print(f"[INFO] Rendering ultra-large: {final_width}x{MASTER_HEIGHT}")
for t in range(len(frames_orig)):
    # 1. Resize del video originale per farlo alto 1080p
    video_frame_large = cv2.resize(frames_orig[t], (video_display_w, MASTER_HEIGHT))
    
    # 2. Render del grafico gigante 1080p
    graph_img = render_massive_graph(keypoints[t], size=GRAPH_WIDTH)
    
    # 3. Unione
    combined = np.hstack((video_frame_large, graph_img))
    out.write(combined)

out.release()
print(f"Fatto! Grafico gigante pronto per la presentazione: {output_video}")