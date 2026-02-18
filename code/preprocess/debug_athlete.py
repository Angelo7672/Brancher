import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- CONFIGURAZIONE PERCORSI ---
base_path = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/data/train_set/S1/"
vid_path = base_path + "Lutz_4_cam_1.mp4"
npy_path = base_path + "Lutz_4_cam_1_h36m.npy"
output_file = "output_skeleton_comparison.mp4"

# --- TOPOLOGIA HUMAN3.6M (Connessioni tra i joint) ---
# Questi sono i "segmenti" che formano lo scheletro
H36M_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

def create_comparison_video_original():
    # 1. Caricamento Dati
    cap = cv2.VideoCapture(vid_path)
    data_3d = np.load(npy_path) # Shape: (217, 17, 3)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Shape Video: ({n_frames}, {height}, {width}, 3)")
    print(f"Shape Notazioni 3D: {data_3d.shape}")

    # Preparazione Video Writer (Larghezza raddoppiata per il side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width * 2, height))

    # Setup Matplotlib per il render 3D
    fig = plt.figure(figsize=(height/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()

    for i in range(min(n_frames, data_3d.shape[0])):
        ret, frame = cap.read()
        if not ret: break

        # Pulizia e plot 3D
        ax.clear()
        # Imposta i limiti (aggiusta in base alla scala dei tuoi dati, es. metri o mm)
        ax.set_xlim3d([-8000, 8000]); ax.set_ylim3d([-8000, 8000]); ax.set_zlim3d([0,1500])
        ax.set_title(f"3D Skeleton - Frame {i}")
        ax.set_xlabel('Asse X (mm)')
        ax.set_ylabel('Asse Y (mm)')
        ax.set_zlabel('Asse Z (mm)')
        ax.view_init(elev=10, azim=180)
        
        points = data_3d[i]
        
        # Disegna le ossa
        for start, end in H36M_BONES:
            ax.plot([points[start, 0], points[end, 0]],
                    [points[start, 1], points[end, 1]],
                    [points[start, 2], points[end, 2]], c='blue', linewidth=2)
        
        # Disegna i joint
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=20)

        # Converte il plot Matplotlib in array immagine
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        plot_img = cv2.resize(plot_img, (width, height))

        # Unione Side-by-Side
        combined = np.hstack((frame, plot_img))
        out.write(combined)
        
        if i % 50 == 0: print(f"Processing frame {i}...")

    cap.release()
    out.release()
    plt.close()
    print(f"✅ Video salvato come: {output_file}")


def create_comparison_video():
    cap = cv2.VideoCapture(vid_path)
    data_3d_raw = np.load(npy_path) # Shape: (217, 17, 3)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width * 2, height))

    fig = plt.figure(figsize=(height/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    for i in range(min(n_frames, data_3d_raw.shape[0])):
        ret, frame = cap.read()
        if not ret: break

        # --- PREPROCESSING REAL-TIME PER VISUALIZZAZIONE ---
        # 1. Scala in metri
        points = data_3d_raw[i].copy().astype(np.float32) / 1000.0
        
        # 2. Swap e Inversione: Athlete(X, Y, Z) -> H36M(X, Z, -Y)
        # Ora: X=destra, Y=altezza, Z=profondità (specchiata)
        x_new = points[:, 0]
        y_new = points[:, 2]      # La vecchia Z diventa l'altezza
        z_new = -points[:, 1]     # La vecchia Y diventa profondità invertita
        
        points_h36m = np.stack([x_new, y_new, z_new], axis=-1)

        # --- PLOT ---
        ax.clear()
        
        # Limiti in metri (ora centrati su 0,0,0 grazie al root centering)
        # Lo skeleton di un atleta è alto circa 1.7-2m, quindi +-1m basta
        ax.set_xlim3d([-4, 4]) 
        ax.set_ylim3d([0, 1.5]) # Altezza (Y in H36M)
        ax.set_zlim3d([-4, 4])
        
        ax.set_title(f"H36M Style (meters) - Frame {i}")
        ax.set_xlabel('X (Destra)')
        ax.set_ylabel('Y (Altezza)')
        ax.set_zlabel('Z (Profondità)')
        
        # Azimut a 0 o 180 per vedere l'atleta di fronte/dietro
        ax.view_init(elev=15, azim=270, vertical_axis='y') 
        
        # Disegna ossa
        for start, end in H36M_BONES:
            ax.plot([points_h36m[start, 0], points_h36m[end, 0]],
                    [points_h36m[start, 1], points_h36m[end, 1]],
                    [points_h36m[start, 2], points_h36m[end, 2]], c='green', linewidth=2)
        
        ax.scatter(points_h36m[:, 0], points_h36m[:, 1], points_h36m[:, 2], c='black', s=15)

        # Rendering Matplotlib -> OpenCV
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        plot_img = cv2.resize(plot_img, (width, height))

        combined = np.hstack((frame, plot_img))
        out.write(combined)
        
        if i % 50 == 0: print(f"Processing frame {i}...")

    cap.release()
    out.release()
    plt.close()
    print(f"✅ Video trasformato salvato: {output_file}")

if __name__ == "__main__":
    create_comparison_video()