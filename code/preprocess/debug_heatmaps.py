import torch
import cv2
from torchvision.io import read_video, write_video
import matplotlib.cm as cm
import torch.nn.functional as F

# === Percorsi ===
video_path = "/Experiments/angeloa00/AirPose/train/block_000/video.mp4"
heatmap_path = "/Experiments/angeloa00/AirPose_COCO/train/block_000/heatmaps.pt"
output_video_path = "video_with_heatmaps.mp4"
output_video_with_kp = "video_with_keypoints.mp4"

# === Carica video e heatmaps ===
video_frames, audio, info = read_video(video_path, output_format="THWC", pts_unit="sec")
video_frames = video_frames.to(torch.float32) / 255.0  # Normalizziamo su [0,1]

heatmaps = torch.load(heatmap_path).to(torch.float32)  # [1, T, J, H, W]
heatmaps = heatmaps.squeeze(0)  # [T, J, H, W]

T, J, H_hmap, W_hmap = heatmaps.shape
print(f"Video frames: {video_frames.shape}, Heatmaps: {heatmaps.shape}")

# === Funzione per colorare e sovrapporre heatmap ===
def overlay_heatmap(frame, hmap, alpha=0.5):
    """
    frame: torch.Tensor [H, W, 3] float32 [0,1]
    hmap: torch.Tensor [J, Hh, Ww] float32
    """
    H, W = frame.shape[:2]

    # Somma tutte le heatmaps dei joints
    hmap_sum = hmap.sum(dim=0, keepdim=True).unsqueeze(0)  # [1,1,Hh,Ww]

    # Resize heatmap a dimensione frame
    hmap_resized = F.interpolate(hmap_sum, size=(H, W), mode='bilinear', align_corners=False)[0,0]  # [H,W]

    # Normalize 0-1
    hmap_norm = (hmap_resized - hmap_resized.min()) / (hmap_resized.max() - hmap_resized.min() + 1e-6)

    # Converti in colore usando cmap 'jet'
    cmap = cm.get_cmap("jet")
    hmap_color = torch.tensor(cmap(hmap_norm.cpu().numpy())[:, :, :3])  # [H,W,3], RGB

    # Sovrapponi
    overlay = alpha * hmap_color + (1-alpha) * frame.cpu().numpy()
    overlay = (overlay * 255).to(torch.uint8)  # resta su GPU
    overlay_bgr = cv2.cvtColor(overlay.cpu().numpy(), cv2.COLOR_RGB2BGR)

    return overlay_bgr

# === Funzione per estrarre keypoints dai heatmaps ===
def extract_keypoints_from_heatmap(hmap):
    """
    hmap: [J, H, W]
    returns: [J, 3] -> (x, y, conf)
    """
    J, H, W = hmap.shape
    keypoints = torch.zeros((J,3), dtype=torch.float32)

    for j in range(J):
        h = hmap[j]
        conf, idx = torch.max(h.view(-1), 0)
        y, x = divmod(int(idx.item()), W)
        keypoints[j] = torch.tensor([x, y, conf])
    return keypoints

# === Funzione per plottare keypoints su frame ===
def draw_keypoints(frame, keypoints, color=(0,0,255), radius=3):
    """
    frame: numpy array [H,W,3] RGB float32 [0,1]
    keypoints: [J,3] tensor (x, y, conf)
    """
    frame_draw = (frame * 255).cpu().numpy().astype('uint8')
    frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR)

    H, W, _ = frame_draw.shape
    for j in range(keypoints.shape[0]):
        x, y, conf = keypoints[j]
        if conf > 0:
            # scala x e y se heatmap è più piccola del video
            x = int(x / W_hmap * W)
            y = int(y / H_hmap * H)
            cv2.circle(frame_draw, (x,y), radius, color, -1)
    return frame_draw

# === Scrittura video con OpenCV ===
fps = int(info['video_fps'])
H_vid, W_vid = video_frames.shape[1:3]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (W_vid, H_vid))

fourccc = cv2.VideoWriter_fourcc(*'mp4v')
out_kp = cv2.VideoWriter(output_video_with_kp, fourccc, fps, (W_vid, H_vid))

for t in range(T):
    frame = video_frames[t]  # [H,W,3], float
    hmap = heatmaps[t]       # [J,H,W]

    overlay = overlay_heatmap(frame, hmap)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    out.write(overlay_bgr)

    keypoints = extract_keypoints_from_heatmap(hmap)
    frame_kp = draw_keypoints(frame, keypoints)
    out_kp.write(frame_kp)

out.release()
out_kp.release()

print(f"Video con heatmap salvato in: {output_video_path}")
print(f"Video con keypoints salvato in: {output_video_with_kp}")
