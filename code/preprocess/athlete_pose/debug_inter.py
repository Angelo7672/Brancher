import torch
import cv2
import numpy as np
import pickle
import os
import os.path as osp
from tqdm import tqdm

# Schema connessioni COCO
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

H36M_SKELETON = [
    [0, 1], [1, 2], [2, 3],       # Gamba destra
    [0, 4], [4, 5], [5, 6],       # Gamba sinistra
    [0, 7], [7, 8], [8, 9], [9, 10], # Tronco e testa
    [8, 11], [11, 12], [12, 13],  # Braccio sinistro
    [8, 14], [14, 15], [15, 16]   # Braccio destro
]

def extract_keypoints(heatmaps, metas, device='cpu'):
    B, J, Hh, Wh = heatmaps.shape 
    heatmaps_flat = heatmaps.view(B, J, -1)
    conf_max, idx_max = torch.max(heatmaps_flat, dim=2)

    # Coordinate locali con correzione half-pixel per precisione
    y_local = (torch.div(idx_max, Wh, rounding_mode='floor')).float() + 0.5
    x_local = (idx_max % Wh).float() + 0.5

    metas_tensor = torch.tensor(metas, device=device, dtype=torch.float32)
    x1, y1 = metas_tensor[:, 1], metas_tensor[:, 2]
    crop_h, crop_w = metas_tensor[:, 3], metas_tensor[:, 4]

    scale_x = (crop_w / Wh)[:, None]
    scale_y = (crop_h / Hh)[:, None]
    
    x_global = x_local * scale_x + x1[:, None]
    y_global = y_local * scale_y + y1[:, None]

    return torch.stack([x_global, y_global, conf_max], dim=2)

def run_debug(num):
    pkl_path = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/train.pkl'
    hmap_path = f'/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/heatmap/train/vid_{num}.pt'
    video_root = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/data/train_set'
    output_video = 'debug_vid_0_with_bbox.mp4'
    
    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)
    
    video_meta = [item for item in db if item['videoid'] == num][:81]
    heatmaps = torch.load(hmap_path, map_location='cpu')[:81]
    
    first_item = video_meta[0]
    subject, subaction = first_item['subject'], first_item['subaction']
    cam_num = first_item['camera_param']['name'].split('_')[-1] 
    vid_file = osp.join(video_root, subject, f"{subaction}_cam_{cam_num}.mp4")
    print(vid_file)
    vid_file = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/data/train_set/S3/Running_55_cam_1.mp4"
    
    # Metadati: [frame_idx, x1, y1, h, w]
    metas_list = [[i, m['box'][0], m['box'][1], m['box'][3]-m['box'][1], m['box'][2]-m['box'][0]] 
                  for i, m in enumerate(video_meta)]
    
    keypoints = extract_keypoints(heatmaps, metas_list)

    cap = cv2.VideoCapture(vid_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if first_item['rotation']: w, h = h, w

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in tqdm(range(len(video_meta))):
        ret, frame = cap.read()
        if not ret: break
        if first_item['rotation']: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # --- DISEGNO BBOX ---
        # m['box'] è [x1, y1, x2, y2]
        box = video_meta[i]['box'].astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(frame, "BBOX", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        pts = keypoints[i].numpy() 

        # --- DISEGNO SKELETON ---
        for connection in COCO_SKELETON:
            p1, p2 = pts[connection[0]], pts[connection[1]]
            if p1[2] > 0.15 and p2[2] > 0.15:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)

        # --- DISEGNO JOINTS ---
        for j, (x, y, conf) in enumerate(pts):
            if conf > 0.15:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Debug salvato: {output_video}")


def coco_to_h36m(keypoints_out):
    """
    Mapping COCO (17 joints) -> H36M (17 joints)
    Input: [T, 17, 3]
    """
    T = keypoints_out.shape[0]
    h36m_kps = torch.zeros((T, 17, 3), device=keypoints_out.device)

    # 0: Pelvis (Centro tra le anche)
    h36m_kps[:, 0, :] = (keypoints_out[:, 11, :] + keypoints_out[:, 12, :]) * 0.5
    # 1: Right Hip / 2: Right Knee / 3: Right Ankle
    h36m_kps[:, 1, :] = keypoints_out[:, 12, :]
    h36m_kps[:, 2, :] = keypoints_out[:, 14, :]
    h36m_kps[:, 3, :] = keypoints_out[:, 16, :]
    # 4: Left Hip / 5: Left Knee / 6: Left Ankle
    h36m_kps[:, 4, :] = keypoints_out[:, 11, :]
    h36m_kps[:, 5, :] = keypoints_out[:, 13, :]
    h36m_kps[:, 6, :] = keypoints_out[:, 15, :]
    # 8: Thorax (Centro spalle)
    h36m_kps[:, 8, :] = (keypoints_out[:, 5, :] + keypoints_out[:, 6, :]) * 0.5
    # 7: Spine (Centro tra Pelvis e Thorax)
    h36m_kps[:, 7, :] = (h36m_kps[:, 0, :] + h36m_kps[:, 8, :]) * 0.5
    # 9: Neck (Naso come proxy) / 10: Head (Centro orecchie)
    h36m_kps[:, 9, :] = keypoints_out[:, 0, :]
    h36m_kps[:, 10, :] = (keypoints_out[:, 1, :] + keypoints_out[:, 2, :]) * 0.5
    # 11, 12, 13: Left Shoulder, Elbow, Wrist
    h36m_kps[:, 11, :] = keypoints_out[:, 5, :]
    h36m_kps[:, 12, :] = keypoints_out[:, 7, :]
    h36m_kps[:, 13, :] = keypoints_out[:, 9, :]
    # 14, 15, 16: Right Shoulder, Elbow, Wrist
    h36m_kps[:, 14, :] = keypoints_out[:, 6, :]
    h36m_kps[:, 15, :] = keypoints_out[:, 8, :]
    h36m_kps[:, 16, :] = keypoints_out[:, 10, :]

    return h36m_kps


def run_comparison_debug(num):
    pkl_path = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/train.pkl'
    hmap_path = f'/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/heatmap/train/vid_{num}.pt'
    video_root = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/data/train_set'
    output_video = 'comparison_2d_3d.mp4'
    
    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)
    
    video_meta = [item for item in db if item['videoid'] == num]
    heatmaps = torch.load(hmap_path, map_location='cpu')
    
    # Prepariamo i metadati per l'estrazione 2D
    metas_list = [[i, m['box'][0], m['box'][1], m['box'][3]-m['box'][1], m['box'][2]-m['box'][0]] 
                  for i, m in enumerate(video_meta)]

    first_item = video_meta[0]
    subject, subaction = first_item['subject'], first_item['subaction']
    cam_num = first_item['camera_param']['name'].split('_')[-1] 
    vid_file = osp.join(video_root, subject, f"{subaction}_cam_{cam_num}.mp4")
    vid_file = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/data/train_set/S3/Running_55_cam_1.mp4"
    
    # Estrazione 2D (Usa la tua funzione extract_keypoints e poi coco_to_h36m)
    kps_coco = extract_keypoints(heatmaps, metas_list)
    kps_2d_h36m = coco_to_h36m(kps_coco).numpy()

    cap = cv2.VideoCapture(vid_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video affiancato (w * 2)
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))

    for i in tqdm(range(len(video_meta))):
        ret, frame = cap.read()
        if not ret: break
        
        # Applica il flip se necessario (come abbiamo discusso prima)
        #frame = cv2.flip(frame, 1) 

        # Creiamo i due pannelli
        pane_2d = frame.copy()
        pane_3d = frame.copy()

        # --- LATO SINISTRA: TUE HEATMAP (2D) ---
        pts_2d = kps_2d_h36m[i]
        for conn in H36M_SKELETON:
            p1, p2 = pts_2d[conn[0]], pts_2d[conn[1]]
            if p1[2] > 0.1: # Confidenza
                cv2.line(pane_2d, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
        cv2.putText(pane_2d, "ESTRATTO DA HEATMAP (2D)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- LATO DESTRA: GROUND TRUTH (3D PROJECTED) ---
        pts_3d = video_meta[i]['joint_3d_image']
        # NOTA: Qui usiamo solo X e Y di joint_3d_image
        for conn in H36M_SKELETON:
            p1, p2 = pts_3d[conn[0]], pts_3d[conn[1]]
            cv2.line(pane_3d, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
        cv2.putText(pane_3d, "GROUND TRUTH (3D PROJ)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Uniamo i frame
        combined = np.hstack((pane_2d, pane_3d))
        out.write(combined)

    cap.release()
    out.release()


def string_debug():
    pkl_path = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/train.pkl'
    
    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)
    
    # Filtriamo solo i frame del video 0
    video_meta = [item for item in db if item['videoid'] == 0]
    
    print(f"Total frames for video 0: {len(video_meta)}")
    print("-" * 30)

    for i in range(min(20, len(video_meta))):
        item = video_meta[i]
        
        # Estraiamo le info principali senza printare gli array enormi
        subject = item['subject']
        subaction = item['subaction']
        cam_name = item['camera_param']['name']
        box = item['box']
        
        print(f"FRAME {i}:")
        print(f"  Subject: {subject} | Subaction: {subaction} | Cam: {cam_name}")
        print(f"  Box [x1, y1, x2, y2]: {box}")
        print(f"  HMap Path: {item['hmap_path']}")
        # Se vuoi vedere solo un pezzetto dei joint senza intasare il terminale:
        print(f"  Joint 3D Image (first joint): {item['joint_3d_image'][0]}") 
        print("-" * 15)

def deep_debug_20_videos():
    pkl_path = '/mnt/proj2/eu-25-52/ar17/AthletePose3D/preprocess/train.pkl'
    
    if not os.path.exists(pkl_path):
        print(f"❌ Errore: Il file {pkl_path} non esiste.")
        return

    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)
    
    seen_videos = set()
    video_data = []

    # Estraiamo il primo frame per ogni video unico
    for item in db:
        vid_id = item['videoid']
        if vid_id not in seen_videos:
            seen_videos.add(vid_id)
            video_data.append(item)
            if len(video_data) >= 20:
                break

    print(f"{'ID':<3} | {'SUBJ':<5} | {'CAMERA':<12} | {'ROT':<5} | {'RES':<10} | {'ACTION':<15}")
    print("-" * 85)

    for item in video_data:
        vid_id = item['videoid']
        subj = item['subject']
        cam_id = item['cameraid']
        # Gestione rotazione (True/False)
        rot = "SI" if item.get('rotation', False) else "NO"
        # Risoluzione
        res = f"{item.get('video_width', '?')}x{item.get('video_height', '?')}"
        action = item['subaction']
        
        # Stampa riga sintetica
        print(f"{vid_id:<3} | {subj:<5} | {cam_id:<12} | {rot:<5} | {res:<10} | {action:<15}")
        
        # Dettagli tecnici sotto ogni riga
        box = item['box']
        h_path = item['hmap_path']
        focal = item['camera_param']['fx']
        
        print(f"   ↳ 📂 H-PATH: {h_path}")
        print(f"   ↳ 📦 BBOX:   [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        print(f"   ↳ 🎥 OPTICS: fx: {focal:.1f} | cx: {item['camera_param']['cx']:.1f} | cy: {item['camera_param']['cy']:.1f}")
        print(f"   ↳ 📍 CENTER: {item['center']}")
        print("-" * 85)

if __name__ == "__main__":
    num = 0
    #deep_debug_20_videos()
    #strin_debug()
    run_debug(num)
    run_comparison_debug(num)