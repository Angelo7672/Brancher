import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
mpl.rcParams['path.simplify'] = False
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['lines.solid_capstyle'] = 'round'
mpl.rcParams['lines.solid_joinstyle'] = 'round'
import os

H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), 
    (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
]

def draw_skeleton(ax, points, color, connections, label):
    ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2], 
        s=40, 
        c=color, 
        edgecolors=color, 
        label=label,
        linewidths=2.5
    )
    for i, j in connections:
        ax.plot(points[[i, j], 0], 
                points[[i, j], 1], 
                points[[i, j], 2], 
                color=color, 
                linewidth=5.0,
                solid_capstyle='round',
                solid_joinstyle='round',
                antialiased=True
        )

def joints3d_to_img(joints_red, joints_blue, size, zoom_level=5.0):
    fig = plt.figure(figsize=(9.5,9.5), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")

    p_red = joints_red.copy()
    p_blue = joints_blue.copy()
    p_red[:, 1] *= -1
    p_blue[:, 1] *= -1

    draw_skeleton(ax, p_red, 'red', H36M_CONNECTIONS, "Posa 1")
    draw_skeleton(ax, p_blue, 'green', H36M_CONNECTIONS, "Posa 2")

    ax.view_init(elev=10, azim=-45, vertical_axis='y') 
    ax.dist = zoom_level 

    all_points = np.vstack([p_red, p_blue])
    mid = all_points.mean(axis=0)
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 1.6

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_axis_on()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    plt.tight_layout()
    canvas.draw()
    
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return cv2.resize(img, size)

def run_comparison(video_path, npy_path_1, npy_path_2, out_path="comparison_3d.mp4"):
    # 1. Caricamento Video in memoria per gestire padding/stride
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRORE] Video non trovato: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    current_frames = len(frames)
    target_frames = 81
    print(f"[INFO] Video caricato: {current_frames} frame")

    # 2. Applicazione Logica Dinamica (Padding / Stride / Troncamento)
    # Questa logica deve rispecchiare ESATTAMENTE quella usata in inferenza
    indices = []
    
    if current_frames < target_frames:
        # Caso 1: Padding (ripetizione dell'ultimo frame)
        indices = list(range(current_frames)) + [current_frames - 1] * (target_frames - current_frames)
        print(f"[INFO] Video corto: Applicato padding a {target_frames}")

    elif current_frames < 2 * target_frames:
        # Caso 2: Troncamento semplice (Stride 1)
        indices = list(range(target_frames))
        print(f"[INFO] Video medio: Troncamento a {target_frames}")

    elif current_frames < 3 * target_frames:
        # Caso 3: Stride 2
        indices = list(range(0, target_frames * 2, 2))
        print(f"[INFO] Video lungo: Stride 2 applicato")

    else:
        # Caso 4: Stride 3
        indices = list(range(0, target_frames * 3, 3))
        print(f"[INFO] Video molto lungo: Stride 3 applicato")

    # 3. Caricamento Pose
    data_red = np.load(npy_path_1)   # (81, 17, 3)
    data_blue = np.load(npy_path_2)  # (81, 17, 3)

    # 4. Scrittura Video Finale
    combined_width = width + height 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (combined_width, height))

    print(f"[INFO] Generazione video di confronto (81 frame totali)...")

    for i in range(target_frames):
        # Recupero il frame corretto basato sull'indice calcolato
        frame_idx = indices[i]
        frame = frames[frame_idx]

        # Generazione immagine 3D
        pose_img = joints3d_to_img(data_red[i], data_blue[i], size=(height, height))

        # Concatenazione
        combined = cv2.hconcat([frame, pose_img])
        out.write(combined)
        
        if i % 10 == 0:
            print(f"Progresso: {i}/{target_frames}...", end='\r')

    out.release()
    print(f"\n[OK] Video salvato: {out_path}")

if __name__ == "__main__":
    PATH_VIDEO = "AR17/code/video/IMG_3990.mp4"
    PATH_NPY_ROSSO = "brancher_inference_coords.npy"    #SEMPRE BRANCHER QUI
    PATH_NPY_BLU = "brancher_self.npy"
    
    run_comparison(PATH_VIDEO, PATH_NPY_ROSSO, PATH_NPY_BLU)