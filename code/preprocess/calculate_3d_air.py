import os
import json
import pickle
import numpy as np
import cv2
import pandas as pd # Usiamo pandas per interpolazione comoda
from scipy.signal import savgol_filter
import shutil # Aggiunto per la copia delle cartelle

# === CONFIGURAZIONE ===
CONF_THRESH = 0.25      # Ignora punti con confidenza 2D sotto
REPROJ_THRESH = 70.0    # Ignora punti 3D con errore di riproiezione > 50 pixel
ROOT_EXP = "/Experiments/angeloa00/AirPose_COCO/test"
MARKER_PATH_TEMPLATE = "/Experiments/angeloa00/AirPose_COCO/{}/markerposes_corrected_all.pkl"
CALIB_PATH_TEMPLATE = "/Experiments/angeloa00/AirPose_COCO/{}/camera_calib.yml"
MACHINE_DIRS = ["machine_1", "machine_2"]
# Variabili per i percorsi di origine e destinazione
SOURCE_BASE = "/Experiments/angeloa00/AirPose"
TARGET_BASE = "/Experiments/angeloa00/AirPose_COCO"


# =========================================================================
# === INIZIALIZZAZIONE: Copia Cartelle Machine da AirPose a AirPose_COCO ===
# =========================================================================
print("=== 📂 Inizializzazione: Copia dei file di Calibrazione e Markers ===")

for machine_dir_name in MACHINE_DIRS:
    source_path = os.path.join(SOURCE_BASE, machine_dir_name)
    target_path = os.path.join(TARGET_BASE, machine_dir_name)
    
    # 1. Crea la cartella di destinazione
    os.makedirs(target_path, exist_ok=True)
    
    # 2. Copia i file di calibrazione e marker
    
    # Copia camera_calib.yml (contiene K scalata e D originale)
    calib_file = "camera_calib.yml"
    src_calib = os.path.join(source_path, calib_file)
    dst_calib = os.path.join(target_path, calib_file)
    if os.path.exists(src_calib):
        shutil.copy2(src_calib, dst_calib)
        print(f"[INFO] Copiato {calib_file} da {source_path} a {target_path}")
    else:
        print(f"[WARN] File {src_calib} non trovato. Necessario per la triangolazione.")
        
    # Copia markerposes_corrected_all.pkl
    marker_file = "markerposes_corrected_all.pkl"
    src_marker = os.path.join(source_path, marker_file)
    dst_marker = os.path.join(target_path, marker_file)
    if os.path.exists(src_marker):
        shutil.copy2(src_marker, dst_marker)
        print(f"[INFO] Copiato {marker_file} da {source_path} a {target_path}")
    else:
        print(f"[WARN] File {src_marker} non trovato.")

print("=== Copia e Setup completati. Inizio elaborazione dei blocchi. ===")
# =========================================================================


def load_calib(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    D = cv_file.getNode("D").mat()
    cv_file.release()
    return K, D

def get_projection_matrix(marker_entry, K):
    # Estrae R, t dal dizionario marker e costruisce P = K @ [R|t]
    cam_key = list(marker_entry.keys())[0]
    rvec = np.array(marker_entry[cam_key]['rvec'])
    tvec = np.array(marker_entry[cam_key]['tvec']).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    # Extrinsics: [R | t]
    Ext = np.hstack([R, tvec])
    P = K @ Ext
    return P, R, tvec

def triangulate_robust(kp1, kp2, K1, D1, K2, D2, P1, P2):
    """
    kp1, kp2: Arrays (N, 3) -> [x, y, conf]
    Ritorna: pts3d (N, 3) con NaN dove la triangolazione fallisce
    """
    # I punti vengono convertiti in float32 e reshape (N, 1, 2) per l'undistort
    pts1_2d = np.ascontiguousarray(kp1[:, :2]).astype(np.float32).reshape(-1, 1, 2)
    pts2_2d = np.ascontiguousarray(kp2[:, :2]).astype(np.float32).reshape(-1, 1, 2)
    
    # 1. UNDISTORT POINTS (Fondamentale!)
    # P=K1 (o K2) assicura che i punti di output siano ancora in coordinate
    # pixel (scalate) anziché in coordinate normalizzate
    pts1_undist = cv2.undistortPoints(pts1_2d, K1, D1, P=K1)
    pts2_undist = cv2.undistortPoints(pts2_2d, K2, D2, P=K2)
    
    # Rimuovi dimensione extra data da undistortPoints
    pts1_u = pts1_undist.reshape(-1, 2)
    pts2_u = pts2_undist.reshape(-1, 2)
    
    # 2. TRIANGOLAZIONE VETTORIZZATA
    pts4d = cv2.triangulatePoints(P1, P2, pts1_u.T, pts2_u.T)
    # Conversione da coordinate omogenee 4D a coordinate 3D
    pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3)
    
    # 3. VERIFICA REPROJECTION ERROR & CONFIDENCE
    final_pts = []
    
    for i in range(len(pts3d)):
        # Check confidenza input
        if kp1[i, 2] < CONF_THRESH or kp2[i, 2] < CONF_THRESH:
            final_pts.append([np.nan]*3)
            continue
            
        X = pts3d[i]
        X_h = np.array([X[0], X[1], X[2], 1.0])
        
        # Proietta indietro su Cam 1
        uv1_proj = P1 @ X_h
        uv1_proj /= uv1_proj[2]
        # L'errore è calcolato rispetto ai punti UNDISTORTI (pts1_u)
        err1 = np.linalg.norm(uv1_proj[:2] - pts1_u[i])
        
        # Proietta indietro su Cam 2
        uv2_proj = P2 @ X_h
        uv2_proj /= uv2_proj[2]
        err2 = np.linalg.norm(uv2_proj[:2] - pts2_u[i])
        
        # Check Reprojection
        if err1 > REPROJ_THRESH or err2 > REPROJ_THRESH:
            final_pts.append([np.nan]*3) # Scarta outlier
        else:
            final_pts.append(X.tolist())
            
    return np.array(final_pts) # (N, 3)

# === CARICAMENTO DATI STATICI ===
# Carica calibrazione (SCALATA) e marker poses (COPIATE)
calibs = {}
markers_dict = {}

for m in MACHINE_DIRS:
    K, D = load_calib(CALIB_PATH_TEMPLATE.format(m))
    calibs[m] = (K, D)
    with open(MARKER_PATH_TEMPLATE.format(m), "rb") as f:
        markers_dict[m] = pickle.load(f)

# === PROCESSING BLOCKS ===
blocks = sorted([d for d in os.listdir(ROOT_EXP) if d.startswith("block_")])

for i in range(0, len(blocks)-1, 2):
    b1, b2 = blocks[i], blocks[i+1]
    print(f"Processing pair: {b1} - {b2}")
    
    path1 = os.path.join(ROOT_EXP, b1, "keypoints2d.json")
    path2 = os.path.join(ROOT_EXP, b2, "keypoints2d.json")
    
    with open(path1) as f: kps1_raw = json.load(f)
    with open(path2) as f: kps2_raw = json.load(f)
    
    # Gestione lista vs dict (assicura che i frame siano indicizzati come stringhe)
    if isinstance(kps1_raw, list): kps1_raw = {str(idx): v for idx, v in enumerate(kps1_raw)}
    if isinstance(kps2_raw, list): kps2_raw = {str(idx): v for idx, v in enumerate(kps2_raw)}
    
    common_frames = sorted(set(kps1_raw.keys()) & set(kps2_raw.keys()), key=lambda x: int(x))
    
    full_sequence_3d = []
    
    for fid in common_frames:
        # Il frame ID per i marker è padded con zeri (es. "007580")
        fid_marker = f"{int(fid):06d}" 
        
        # Check esistenza marker
        if fid_marker not in markers_dict["machine_1"] or fid_marker not in markers_dict["machine_2"]:
            full_sequence_3d.append([[np.nan]*3] * 17) # Assuming 17 joints (COCO), riempie con NaN
            continue
            
        # Get Matrix P1
        K1, D1 = calibs["machine_1"]
        P1, _, _ = get_projection_matrix(markers_dict["machine_1"][fid_marker], K1)
        
        # Get Matrix P2
        K2, D2 = calibs["machine_2"]
        P2, _, _ = get_projection_matrix(markers_dict["machine_2"][fid_marker], K2)
        
        # Keypoints (N, 3) -> x, y, conf
        pts1 = np.array(kps1_raw[fid]) # shape (N, 3) assumed
        pts2 = np.array(kps2_raw[fid])
        
        # Triangola frame corrente
        pts3d_frame = triangulate_robust(pts1, pts2, K1, D1, K2, D2, P1, P2)
        full_sequence_3d.append(pts3d_frame)
    
    # === POST-PROCESSING (Interpolazione + Smoothing) ===
    # Convertiamo in numpy (T, N_joints, 3)
    seq_np = np.array(full_sequence_3d) # T x N x 3
    T, N, C = seq_np.shape
    
    cleaned_seq = []
    
    # Iteriamo per ogni joint
    for j in range(N):
        # Converti traiettoria joint j in DataFrame per interpolare facile
        traj = pd.DataFrame(seq_np[:, j, :], columns=['x', 'y', 'z'])
        
        # 1. Interpola i NaN (spline con fallback lineare)
        # Controlliamo prima il numero di punti non-NaN
        valid_count = traj.dropna().shape[0]
        if valid_count > 4: # Se ci sono abbastanza punti per una spline di ordine 3 (4 minimi)
            try:
                # Tenta interpolazione spline
                traj_interp = traj.interpolate(method='spline', order=3, limit_direction='both')
            except ValueError as e:
                # L'errore dfitpack è spesso mascherato come ValueError in Python
                print(f"[WARN] Spline failed for joint {j} (valid={valid_count}). Falling back to linear interpolation.")
                traj_interp = traj.interpolate(method='linear', limit_direction='both')
        else:
            # Punti insufficienti, usa interpolazione lineare
            traj_interp = traj.interpolate(method='linear', limit_direction='both')
        
        # 2. Forward Fill e Backward Fill per evitare 0.0 sui bordi (propaga ultimo/primo valido)
        traj_interp = traj_interp.ffill().bfill()
        
        # 3. Se rimangono ANCORA NaN (significa che TUTTA la sequenza è NaN), mettiamo 0.0
        traj_interp = traj_interp.fillna(0.0) 
        
        data_to_smooth = traj_interp.values
        
        # 4. Applica Savitzky-Golay solo se abbiamo abbastanza dati
        if T > 7:
            smoothed_col = np.empty_like(data_to_smooth)
            # Applicazione del filtro su ogni dimensione (x, y, z)
            for c in range(3):
                smoothed_col[:, c] = savgol_filter(data_to_smooth[:, c], window_length=7, polyorder=3)
            cleaned_seq.append(smoothed_col)
        else:
            cleaned_seq.append(data_to_smooth)
            
    # Reshape finale a (T, N, 3) -> trasposto perché il loop era su N
    cleaned_seq = np.array(cleaned_seq).transpose(1, 0, 2)
    
    # Salva Output (stesso file per i due blocchi consecutivi che formano una singola sequenza)
    out_list = cleaned_seq.tolist()
    
    out1 = os.path.join(ROOT_EXP, b1, "keypoints3d.json")
    out2 = os.path.join(ROOT_EXP, b2, "keypoints3d.json")
    
    with open(out1, "w") as f: json.dump(out_list, f)
    with open(out2, "w") as f: json.dump(out_list, f)
    
    print(f"Saved refined 3D poses for {b1}/{b2}")

print("Elaborazione Completata.")