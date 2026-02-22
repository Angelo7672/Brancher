import cv2
import os
from pathlib import Path

# Configurazione
TARGET_WIDTH = 640
TARGET_HEIGHT = 352
FRAMES_PER_CHUNK = 81
TARGET_FPS = 30.0

def process_airpose_dataset(root_dir, output_dir):
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cerchiamo tutti i file "video.mp4" dentro train e test
    # rglob("video.mp4") cerca ricorsivamente in tutte le sottocartelle
    video_paths = sorted(list(root_path.rglob("video.mp4")))

    if not video_paths:
        print(f"Nessun file video.mp4 trovato in {root_dir}")
        return

    print(f"Trovati {len(video_paths)} video originali. Inizio lo split...")

    global_counter = 0

    for v_path in video_paths:
        print(f"Elaborazione: {v_path.relative_to(root_path)}")
        
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            print(f"Errore nell'aprire {v_path}")
            continue

        # Poiché sappiamo che sono 243 frame, faremo 3 cicli da 81
        for chunk_idx in range(3):
            output_filename = f"video_{global_counter:05d}.mp4"
            target_file = output_path / output_filename
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(target_file), 
                fourcc, 
                TARGET_FPS, 
                (TARGET_WIDTH, TARGET_HEIGHT)
            )

            frames_captured = 0
            while frames_captured < FRAMES_PER_CHUNK:
                ret, frame = cap.read()
                if not ret:
                    break
                
                writer.write(frame)
                frames_captured += 1
            
            writer.release()
            global_counter += 1
            
        cap.release()

    print(f"\nFatto! Creati {global_counter} video da 81 frame in: {output_dir}")

if __name__ == "__main__":
    # Assicurati che il percorso 'AirPose' sia corretto rispetto a dove lanci lo script
    percorso_dataset = "/mnt/proj2/eu-25-52/ar17/AirPose" 
    cartella_destinazione = "/mnt/proj2/eu-25-52/ar17/AirPose_H36M"

    process_airpose_dataset(percorso_dataset, cartella_destinazione)