import cv2
import os
from pathlib import Path

FRAMES_PER_VIDEO = 243

def split_videos(
    input_dir,
    output_dir,
    frames_per_video=FRAMES_PER_VIDEO
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.glob("*.mp4"))

    if not video_files:
        print("Nessun file mp4 trovato.")
        return

    t = 0
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Errore nell'apertura di {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        total_frames = len(frames)
        print(f"  Totale frame: {total_frames}")

        num_chunks = (total_frames + frames_per_video - 1) // frames_per_video

        for i in range(num_chunks):
            start = i * frames_per_video
            end = min(start + frames_per_video, total_frames)
            chunk = frames[start:end]

            output_name = f"video_{t}_part{i:03d}.mp4"
            output_path = output_dir / output_name
            t += 1

            # Codec MP4 leggero
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )

            for frame in chunk:
                writer.write(frame)

            writer.release()
            print(f"  Salvato: {output_name} ({len(chunk)} frame)")

    print("\nDone.")


if __name__ == "__main__":
    input_folder = "input_videos"
    output_folder = "output_videos"

    split_videos(input_folder, output_folder)