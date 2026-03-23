# Preprocess our drone's video

import cv2
import os
from pathlib import Path

FRAMES_PER_VIDEO = 81
TARGET_WIDTH = 640
TARGET_HEIGHT = 352
TARGET_FPS = 30.0

def split_videos(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.glob("*.mp4"))

    if not video_files:
        print("No mp4 files found.")
        return

    video_counter = 0

    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening {video_path}")
            continue

        total_frames_in_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_in_src < FRAMES_PER_VIDEO:
            print(f"  Rejected: Video too short ({total_frames_in_src} frames)")
            cap.release()
            continue

        chunk_idx = 0
        while True:
            # Temporary buffer for the frames of the current chunk
            chunk_frames = []
            valid_chunk = True
            
            for _ in range(FRAMES_PER_VIDEO):
                ret, frame = cap.read()
                if not ret:
                    valid_chunk = False # End of video or missing frames to complete the block
                    break
                
                resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                chunk_frames.append(resized_frame)

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
                if not valid_chunk:
                    print(f"  End of video: Advancing frames discarded.")
                break

        video_counter += 1
        cap.release()

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    input_folder = "/mnt/proj2/eu-25-52/ar17/Drone"
    output_folder = "/mnt/proj2/eu-25-52/ar17/DroneDataset/video"

    split_videos(input_folder, output_folder)