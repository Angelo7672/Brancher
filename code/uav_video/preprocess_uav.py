import cv2
import os
from pathlib import Path

# Configurazione costanti
FRAMES_PER_VIDEO = 81
TARGET_WIDTH = 640
TARGET_HEIGHT = 352
TARGET_FPS = 30.0

def split_videos(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cerchiamo tutti i file .mp4
    video_files = sorted(input_dir.glob("*.mp4"))

    if not video_files:
        print("Nessun file mp4 trovato.")
        return

    video_counter = 0 # Contatore globale per i nomi dei file in uscita

    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Errore nell'apertura di {video_path}")
            continue

        # Verifichiamo se il video ha almeno 81 frame
        total_frames_in_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_in_src < FRAMES_PER_VIDEO:
            print(f"  Scartato: video troppo corto ({total_frames_in_src} frame)")
            cap.release()
            continue

        chunk_idx = 0
        while True:
            # Buffer temporaneo per i frame del chunk attuale
            chunk_frames = []
            valid_chunk = True
            
            for _ in range(FRAMES_PER_VIDEO):
                ret, frame = cap.read()
                if not ret:
                    valid_chunk = False # Fine del video o frame mancanti per completare il blocco
                    break
                
                # Ridimensionamento immediato
                resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                chunk_frames.append(resized_frame)

            # Se abbiamo raccolto esattamente 81 frame, salviamo il video
            if valid_chunk and len(chunk_frames) == FRAMES_PER_VIDEO:
                output_name = f"video_{video_counter}_part{chunk_idx:03d}.mp4"
                output_path = output_dir / output_name
                
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    TARGET_FPS,
                    (TARGET_WIDTH, TARGET_HEIGHT)
                )

                for f in chunk_frames:
                    writer.write(f)
                
                writer.release()
                print(f"  Salvato: {output_name}")
                chunk_idx += 1
            else:
                # Se il chunk non è completo (meno di 81), usciamo dal loop del video
                if not valid_chunk:
                    print(f"  Fine video: frame avanzanti scartati.")
                break

        video_counter += 1
        cap.release()

    print("\nProcesso completato con successo.")

if __name__ == "__main__":
    # Assicurati che queste cartelle esistano o vengano create
    input_folder = "/mnt/proj2/eu-25-52/ar17/Drone"
    output_folder = "/mnt/proj2/eu-25-52/ar17/DroneDataset/video"

    split_videos(input_folder, output_folder)