import cv2
import shutil
from pathlib import Path

def flip_and_augment_videos(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prendiamo tutti i file .mp4
    video_files = sorted(list(input_dir.glob("*.mp4")))
    
    if not video_files:
        print("Nessun file mp4 trovato nella cartella di input.")
        return

    num_originals = len(video_files)
    print(f"Trovati {num_originals} video. Inizio elaborazione...")

    # --- FASE 1: Copia degli originali con nuovo nome ---
    print("\nCopia dei video originali...")
    for i, video_path in enumerate(video_files):
        # Nome progressivo: video_000.mp4, video_001.mp4, ecc.
        new_name = f"video_{i:04d}.mp4"
        shutil.copy(video_path, output_dir / new_name)
        if i % 10 == 0: print(f"  Copiati {i}/{num_originals}...")

    # --- FASE 2: Creazione dei video flippati ---
    print("\nCreazione dei video flippati (Horizontal Flip)...")
    for i, video_path in enumerate(video_files):
        # Il nome continua dopo gli originali
        new_name = f"video_{i + num_originals:04d}.mp4"
        output_path = output_dir / new_name
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip orizzontale (1 = orizzontale, 0 = verticale, -1 = entrambi)
            flipped_frame = cv2.flip(frame, 1)
            writer.write(flipped_frame)

        cap.release()
        writer.release()
        
        if i % 10 == 0: 
            print(f"  Elaborati {i}/{num_originals} flippati...")

    print(f"\nOperazione completata! Totale video nella cartella: {num_originals * 2}")

if __name__ == "__main__":
    # Cambia i nomi delle cartelle se necessario
    cartella_sorgente = "/mnt/proj2/eu-25-52/ar17/AirPose_H36M"  # Quella creata dallo script precedente
    cartella_destinazione = "/mnt/proj2/eu-25-52/ar17/AirPose_H36M/video_and_flipped"

    flip_and_augment_videos(cartella_sorgente, cartella_destinazione)