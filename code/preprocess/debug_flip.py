import torch
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Stesse coppie usate per il flip per disegnare le connessioni
KINEMATIC_TREE = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 3], [6, 4], [3, 1], [4, 2], [1, 0], [2, 0]]

def create_comparison_video_hm(src_hm_path, dst_hm_path, out_path, fps=30):
    # Carichiamo e assicuriamoci che siano float32 su CPU
    hm_src = torch.load(src_hm_path).detach().cpu().float().numpy() # [T, J, H, W]
    hm_dst = torch.load(dst_hm_path).detach().cpu().float().numpy()
    T, J, H, W = hm_src.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Nota: W*2 perché mettiamo i video fianco a fianco
    video = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))

    for t in range(T):
        # Prendiamo il massimo valore tra i canali dei joint
        # .copy() assicura che l'array sia contiguo in memoria per OpenCV
        src_frame = np.max(hm_src[t], axis=0).copy() 
        dst_frame = np.max(hm_dst[t], axis=0).copy()
        
        # Gestione valori estremi (NaN o Inf) che mandano in crash minMaxIdx
        src_frame = np.nan_to_num(src_frame)
        dst_frame = np.nan_to_num(dst_frame)

        # Normalizzazione manuale o correzione OpenCV
        def to_img(frame):
            # Normalizziamo in scala 0-255
            f_min, f_max = frame.min(), frame.max()
            if f_max - f_min > 1e-5:
                frame = (frame - f_min) / (f_max - f_min) * 255
            else:
                frame = frame * 0
            return frame.astype(np.uint8)

        src_img = to_img(src_frame)
        dst_img = to_img(dst_frame)
        
        # Applichiamo colormap
        src_color = cv2.applyColorMap(src_img, cv2.COLORMAP_JET)
        dst_color = cv2.applyColorMap(dst_img, cv2.COLORMAP_JET)
        
        combined = np.hstack((src_color, dst_color))
        
        # Scrittura testi informativa
        cv2.putText(combined, f"ORIG - Frame {t}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, f"FLIP", (W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        video.write(combined)
        
    video.release()

def draw_skeleton_2d(frame_kpts, w, h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, j in KINEMATIC_TREE:
        p1 = (int(frame_kpts[i][0]), int(frame_kpts[i][1]))
        p2 = (int(frame_kpts[j][0]), int(frame_kpts[j][1]))
        cv2.line(img, p1, p2, (0, 255, 0), 2)
    for p in frame_kpts:
        cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
    return img

def create_comparison_video_2d(src_json, dst_json, out_path, w, h, fps=30):
    with open(src_json, 'r') as f: k2d_src = json.load(f)
    with open(dst_json, 'r') as f: k2d_dst = json.load(f)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

    for t in range(len(k2d_src)):
        img_src = draw_skeleton_2d(k2d_src[t], w, h)
        img_dst = draw_skeleton_2d(k2d_dst[t], w, h)
        combined = np.hstack((img_src, img_dst))
        video.write(combined)
    video.release()

def create_comparison_video_3d(src_json, dst_json, out_path, fps=30):
    with open(src_json, 'r') as f: k3d_src = json.load(f)
    with open(dst_json, 'r') as f: k3d_dst = json.load(f)
    
    fig = plt.figure(figsize=(10, 5))
    canvas = FigureCanvasAgg(fig)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = None

    for t in range(len(k3d_src)):
        fig.clf()
        for i, data in enumerate([k3d_src[t], k3d_dst[t]]):
            ax = fig.add_subplot(1, 2, i+1, projection='3d')
            pts = np.array(data)
            ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c='r') # Re-mapping assi per visualizzazione
            for edge in KINEMATIC_TREE:
                ax.plot(pts[edge, 0], pts[edge, 2], -pts[edge, 1], color='blue')
            ax.set_title("Originale" if i==0 else "Flipped")
            #ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if video is None:
            h, w, _ = img.shape
            video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        video.write(img)
    video.release()
    plt.close()

# --- ESECUZIONE DEBUG PER UNA CARTELLA ---
sample_folder = "block_009" # Cambia con una cartella esistente
src_path = f"/Experiments/angeloa00/AirPose_COCO/train/{sample_folder}"
dst_path = f"/Experiments/angeloa00/AirPose_COCO/train_flipped/{sample_folder}"
debug_dir = "./debug_output"
os.makedirs(debug_dir, exist_ok=True)

print("Generando video di debug...")
create_comparison_video_hm(os.path.join(src_path, "heatmaps.pt"), os.path.join(dst_path, "heatmaps.pt"), f"{debug_dir}/debug_hm.mp4")
create_comparison_video_2d(os.path.join(src_path, "keypoints2d.json"), os.path.join(dst_path, "keypoints2d.json"), f"{debug_dir}/debug_2d.mp4", 640, 352)
create_comparison_video_3d(os.path.join(src_path, "keypoints3d.json"), os.path.join(dst_path, "keypoints3d.json"), f"{debug_dir}/debug_3d.mp4")
print(f"Video salvati in {debug_dir}")