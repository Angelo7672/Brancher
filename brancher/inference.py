import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
mpl.rcParams['path.simplify'] = False
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['lines.solid_capstyle'] = 'round'
mpl.rcParams['lines.solid_joinstyle'] = 'round'

from config import Config
from model import create_model
from learning import load_weights


H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]


def joints3d_to_img(joints3d, size, flip_y=True):
    fig = plt.figure(figsize=(9.5, 9.5), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")

    points = joints3d.copy()
    if flip_y:
        points[:, 1] *= -1

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color='blue',
        s=40,
        edgecolors='blue',
        linewidths=2.5
    )

    for i, j in H36M_CONNECTIONS:
        ax.plot(
            points[[i, j], 0],
            points[[i, j], 1],
            points[[i, j], 2],
            color='red',
            linewidth=5.0,
            solid_capstyle='round',
            solid_joinstyle='round',
            antialiased=True
        )

    ax.view_init(elev=10, azim=-45, vertical_axis='y')

    mid = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 1.6

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_axis_on()

    ax.set_xlabel('X', labelpad=-10)
    ax.set_ylabel('Y', labelpad=-8)
    ax.set_zlabel('Z', labelpad=-10)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))

    plt.tight_layout()

    canvas.draw()

    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)

    plt.close(fig)

    return cv2.resize(img, size)


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    if torch.is_tensor(data):
        flipped_data = data.clone()
    else:
        flipped_data = copy.deepcopy(data)

    flipped_data[..., 0] *= -1  
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]

    return flipped_data


def run_inference(video_path, checkpoint_path, out_path="inference.mp4"):
    cfg = Config()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Not possible to open the video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_resolution = 640 
    scale = target_resolution / max(width, height)
    new_width = int(round((width * scale) / 32) * 32)
    new_height = int(round((height * scale) / 32) * 32)
    
    print(f"[INFO] Original Dimension: {width}x{height}")
    print(f"[INFO] New Dimensions for Inference: {new_width}x{new_height}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"[INFO] Video loaded: {len(frames)} frames")

    frames_resized = [cv2.resize(f, (new_width, new_height)) for f in frames]
    frames_batch = np.array(frames_resized)
    frames_batch = torch.from_numpy(frames_batch).float().unsqueeze(0)  # (1, T, H, W, 3)
    print(f"[INFO] Batch prepared: {frames_batch.shape}")

    target_frames = 81
    current_frames = frames_batch.shape[1]

    if current_frames < target_frames:
        # --- CASE 1: Very short video -> Padding ---
        padding_size = target_frames - current_frames
        last_frame = frames_batch[:, -1:, :, :, :]
        padding = last_frame.repeat(1, padding_size, 1, 1, 1)
        frames_batch = torch.cat([frames_batch, padding], dim=1)
        applied_stride = 1
        print(f"[INFO] Short video: Padding by {padding_size}")
    elif current_frames < 2 * target_frames:
        # --- CASE 2: Between 81 and 161 frames -> Truncation (Stride 1) ---
        frames_batch = frames_batch[:, :target_frames, :, :, :]
        applied_stride = 1
        print(f"[INFO] Medium Video: Simple Truncation (Stride 1)")
    elif current_frames < 3 * target_frames:
        # --- CASE 3: Between 162 and 242 frames -> Stride 2 ---
        frames_batch = frames_batch[:, :target_frames * 2:2, :, :, :]
        applied_stride = 2
        print(f"[INFO] Long video: Stride 2 applied")
    else:
        # --- CASE 4: Over 243 frames -> Stride 3 ---
        frames_batch = frames_batch[:, :target_frames * 3:3, :, :, :]
        applied_stride = 3
        print(f"[INFO] Very long video: Stride 3 applied")

    print(f"[INFO] Final batch ready: {frames_batch.shape}")


    print("[INFO] Extracting 3D Pose...")
    model = create_model(frame_size = (new_height, new_width), inference=True)
    model = load_weights(model, checkpoint_path, cfg.device)
    model = model.to(cfg.device)
    model.eval()

    with torch.no_grad():
        pred_orig, _, _, _ = model(frames_batch)
        frames_flipped = torch.flip(frames_batch, dims=[3]) 
        pred_flipped_raw, _, _, _ = model(frames_flipped)
        pred_flipped_corrected = flip_data(pred_flipped_raw)
        preds_3d_tensor = (pred_orig + pred_flipped_corrected) / 2
        
    preds_3d = preds_3d_tensor[0].cpu().numpy() # (T, 17, 3)

    npy_out_path = out_path.replace(".mp4", "_coords.npy")
    np.save(npy_out_path, preds_3d)
    print(f"[INFO] 3D Coordinates saved in: {npy_out_path}")


    print("[INFO] Writing Video...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    final_combined_width = new_width * 2
    final_combined_height = new_height
    out = cv2.VideoWriter(out_path, fourcc, fps, (final_combined_width, final_combined_height))

    for i in range(target_frames):
        idx_originale = min(i * applied_stride, len(frames) - 1)
        
        frame_to_show = frames_resized[idx_originale]
        
        joints = preds_3d[i]
        
        pose_img = joints3d_to_img(joints, size=(new_width, new_height))
        pose_img_resized = cv2.resize(pose_img, (new_width, new_height))

        combined = cv2.hconcat([frame_to_show, pose_img_resized])
        
        out.write(combined)

    out.release()
    print(f"[INFO] Video Saved: {out_path}")


if __name__ == "__main__":
    weights_path = "experiments/brancher/FINAL_model_20260321-225014/final_weight_epoch60.pth"
    video_path = "AR17/code/video/IMG_4277.mp4"
    run_inference(video_path, weights_path, out_path="brancher_inference.mp4")