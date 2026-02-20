from keypoint_extractor import KeypointExtractorSeq
import cv2
import numpy as np
import torch
from config import Config

import matplotlib
matplotlib.use('Agg')  # Forza il backend non interattivo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

cfg = Config()
video_path = "/mnt/proj2/eu-25-52/ar17/SportsPose_reprocessed_v2/sample_00350_view_back/video.mp4"
keypoint_extractor = KeypointExtractorSeq(num_joints=17, frame_size=(352, 640), device=cfg.device)


# -----------------------------------
# HUMAN3.6M SKELETON
# -----------------------------------
H36M_BONES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16)
]

def draw_skeleton(frame, keypoints, conf_thresh=0.3):
    """
    keypoints: (17,3) -> x,y,confidence
    """
    for i, (x,y,c) in enumerate(keypoints):
        if c > conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)

    for i,j in H36M_BONES:
        if keypoints[i][2] > conf_thresh and keypoints[j][2] > conf_thresh:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, (0,0,255), 2)

    return frame


def render_skeleton_plt(keypoints, aspect_ratio, output_size=(600, 600)):
    """
    Crea il plot e forza la dimensione dell'immagine in uscita.
    """
    # Usiamo una DPI fissa per controllare meglio la dimensione finale
    fig = plt.figure(figsize=(output_size[0]/100, output_size[1]/100), dpi=100)
    ax = fig.add_subplot(111)
    
    # Range normalizzato
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-aspect_ratio - 0.1, aspect_ratio + 0.1)
    
    ax.invert_yaxis() # Testa in alto
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Disegno ossa H36M
    for i, j in H36M_BONES:
        if keypoints[i, 2] > 0.1 and keypoints[j, 2] > 0.1:
            ax.plot([keypoints[i, 0], keypoints[j, 0]], 
                    [keypoints[i, 1], keypoints[j, 1]], 'r-', linewidth=2)

    ax.scatter(keypoints[:, 0], keypoints[:, 1], s=15, c='green')

    # Estrazione dell'immagine dal buffer
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba_buffer = canvas.buffer_rgba()
    img = np.asarray(rgba_buffer)[:, :, :3] # Prendi solo RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Converti per OpenCV
    
    # FORZA il resize alla dimensione del VideoWriter per sicurezza
    img = cv2.resize(img, output_size)
    
    plt.close(fig)
    return img



# Carica video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERRORE] Impossibile aprire il video: {video_path}")
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

frames_resized = [cv2.resize(f, (640,352)) for f in frames]
frames_batch = np.array(frames_resized)
frames_batch = torch.from_numpy(frames_batch).float().unsqueeze(0)

kpt_tensor = keypoint_extractor(frames_batch)

print("DEBUG",kpt_tensor.shape)

# kpt_tensor shape atteso: (1, T, 17, 3)
kpt_tensor = kpt_tensor.squeeze(0).cpu().numpy()  # (T,17,3)

# Video writer
#out = cv2.VideoWriter(
#    "output_h36m.mp4",
#    cv2.VideoWriter_fourcc(*'mp4v'),
#    fps,
#    (416,640)
#)
#
#for t in range(len(frames_resized)):
#    frame = frames_resized[t].copy()
#
#    keypoints = kpt_tensor[t]
#
#    frame = draw_skeleton(frame, keypoints)
#
#    out.write(frame)
#out.release()
#print(f"[OK] Video salvato correttamente: {output_filename}")

# --- Nuovo loop di salvataggio ---
# --- Configurazione VideoWriter ---
output_filename = "output_cartesian_2d.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Prova 'XVID' se 'mp4v' fallisce
plot_size = (600, 600)
out_cartesian = cv2.VideoWriter(output_filename, fourcc, fps, plot_size)

if not out_cartesian.isOpened():
    print("[ERRORE] VideoWriter non aperto. Controlla i codec.")

ar = 640 / 416

print(f"[INFO] Inizio scrittura {len(kpt_tensor)} frame...")
for t in range(len(kpt_tensor)):
    # Genera il frame del grafico
    cartesian_frame = render_skeleton_plt(kpt_tensor[t], ar, output_size=plot_size)
    
    # Verifica finale dimensioni (se fallisce qui, OpenCV non scrive)
    if cartesian_frame.shape[1] != plot_size[0] or cartesian_frame.shape[0] != plot_size[1]:
        cartesian_frame = cv2.resize(cartesian_frame, plot_size)
        
    out_cartesian.write(cartesian_frame)

out_cartesian.release()
print(f"[OK] Video salvato correttamente: {output_filename}")
