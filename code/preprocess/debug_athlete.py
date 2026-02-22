import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- CONFIGURAZIONE PERCORSI ---
PKL_PATH2 = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/test/frame_81/test/00000002.pkl' 
OUTPUT_NAME = 'debug_2d_3d_comparison.mp4'
FPS = 20

def get_h36m_edges():
    return [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), (8,9), (9,10), 
            (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]

def render_3d_panel(joints3d, edges, size=(640, 480)):
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')

    points = joints3d.copy()
    points[:, 1] *= -1  # Invertiamo Y per orientamento Matplotlib

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=20)
    for start, end in edges:
        ax.plot(points[[start, end], 0], points[[start, end], 1], points[[start, end], 2], color='blue', linewidth=2)

    # Tuning Assi
    mid = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 1.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.view_init(elev=15, azim=-45, vertical_axis='y')

    plt.tight_layout()
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(size[1], size[0], 4)
    plt.close(fig)
    return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)

def draw_skeleton_2d(canvas, joints, edges, color=(0, 255, 0), label=""):
    for start, end in edges:
        p1 = (int(joints[start, 0]), int(joints[start, 1]))
        p2 = (int(joints[end, 0]), int(joints[end, 1]))
        cv2.line(canvas, p1, p2, color, 2)
    cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main():
    if not os.path.exists(PKL_PATH2):
        print(f"❌ File non trovato: {PKL_PATH2}")
        return

    with open(PKL_PATH2, 'rb') as f:
        data_dict2 = pickle.load(f)
    
    inputs = data_dict2['data_input']   # [81, 17, 3]
    labels = data_dict2['data_label']   # [81, 17, 3]

    w_panel, h_panel = 640, 480
    # Video a 2 pannelli (640*2 = 1280px di larghezza)
    out = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w_panel * 2, h_panel))
    edges = get_h36m_edges()

    print(f"🎬 Generazione video (2D + 3D): {OUTPUT_NAME}")

    for i in range(inputs.shape[0]):
        # --- PANNELLO 1: INPUT 2D ---
        panel_2d = np.zeros((h_panel, w_panel, 3), dtype=np.uint8)
        # Normalizzazione coordinate da [-1, 1] a pixel
        pts_2d = (inputs[i, :, :2] + 1) / 2 * np.array([w_panel, h_panel])
        draw_skeleton_2d(panel_2d, pts_2d, edges, (0, 255, 0), "Input 2D")

        # --- PANNELLO 2: RENDERING 3D ---
        panel_3d = render_3d_panel(labels[i], edges, size=(w_panel, h_panel))

        # Unione orizzontale
        combined = np.hstack([panel_2d, panel_3d])
        out.write(combined)
        
        if i % 10 == 0: 
            print(f"Elaborazione frame {i}/{inputs.shape[0]}...")

    out.release()
    print(f"✅ Completato: {OUTPUT_NAME}")

if __name__ == "__main__":
    main()