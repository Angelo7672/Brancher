# Horizontal Flip of the video

import cv2
import shutil
from pathlib import Path

def flip_and_augment_videos(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(input_dir.glob("*.mp4")))
    
    if not video_files:
        print("No mp4 files found in the input folder.")
        return

    num_originals = len(video_files)
    print(f"Found {num_originals} videos. Processing begins...")

    # --- PHASE 1: Copy the originals with the new name ---
    print("\nCopy of the original videos...")
    for i, video_path in enumerate(video_files):
        new_name = f"video_{i:04d}.mp4"
        shutil.copy(video_path, output_dir / new_name)
        if i % 10 == 0: print(f"  Copied {i}/{num_originals}...")

    # --- PHASE 2: Creating the flipped videos ---
    print("\nCreating Flipped Videos (Horizontal Flip)...")
    for i, video_path in enumerate(video_files):
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
            
            # Horizontal Flip
            flipped_frame = cv2.flip(frame, 1)
            writer.write(flipped_frame)

        cap.release()
        writer.release()
        
        if i % 10 == 0: print(f"  Elaborated {i}/{num_originals} flipped...")

    print(f"\nOperation completed! Total videos in the folder: {num_originals * 2}")


if __name__ == "__main__":
    cartella_sorgente = "/mnt/proj2/eu-25-52/ar17/AirPose_H36M"
    cartella_destinazione = "/mnt/proj2/eu-25-52/ar17/AirPose_H36M/video_and_flipped"

    flip_and_augment_videos(cartella_sorgente, cartella_destinazione)