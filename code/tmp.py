import cv2

# === INPUT ===
input_image = "input.jpg"
output_video = "output.mp4"
num_frames = 81
fps = 25  # puoi cambiarlo

# === Leggi immagine ===
img = cv2.imread(input_image)

if img is None:
    raise ValueError(f"Impossibile leggere l'immagine: {input_image}")

height, width = img.shape[:2]

# === Setup VideoWriter ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === Scrivi 81 frame identici ===
for _ in range(num_frames):
    out.write(img)

out.release()

print(f"Video salvato in: {output_video}")