import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def debug_pose_2d(file_path):
    if not os.path.exists(file_path):
        print(f"Errore: File {file_path} non trovato.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Estrazione input (j2d)
    j2d = data['data_input']
    
    # Controllo shape: ci aspettiamo [Joints, 2] oppure [Frames, Joints, 2]
    print(f"--- Debug Info ---")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Shape originale: {j2d.shape}")
    
    # Se j2d è un tensore PyTorch, lo convertiamo in numpy
    if hasattr(j2d, 'numpy'):
        j2d = j2d.numpy()

    # Gestione sequenza vs frame singolo
    if len(j2d.shape) == 3:
        print(f"Rilevata sequenza di {j2d.shape[0]} frame.")
        # Prendiamo il frame centrale per il debug
        frame_idx = j2d.shape[0] // 2
        points = j2d[frame_idx]
        title = f"Frame {frame_idx} (Sequenza)"
    else:
        points = j2d
        title = "Singolo Frame"

    plt.figure(figsize=(8, 8))
    
    # Plot dei punti
    x = points[:, 0]
    y = points[:, 1]
    
    plt.scatter(x, y, c='red', label='Joints')
    
    # Numerazione dei joint per capire l'ordine (es. 0=bacino, 1=anca...)
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), fontsize=9, color='blue', ha='right')

    # Configurazione Assi - CRUCIALE PER IL DEBUG
    plt.xlabel('Asse X')
    plt.ylabel('Asse Y')
    plt.title(f"Visualizzazione 2D: {title}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')  # Mantiene le proporzioni reali
    
    # Mostra i limiti attuali per capire il range (es. [0,1] o pixel)
    print(f"Range X: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Range Y: [{y.min():.2f}, {y.max():.2f}]")
    
    plt.legend()
    plt.show()

# --- ESECUZIONE ---
path_file = "/mnt/proj2/eu-25-52/ar17/AthletePose3D/pose_3d_v3/frame_81/train/00000000.pkl"
debug_pose_2d(path_file)
