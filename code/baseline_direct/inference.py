import torch
import cv2
import numpy as np
from config import Config
from model import Direct3DPoseNet
from utils import set_seed


# -------------------------------
# Carica modello e pesi
# -------------------------------
def load_model(cfg, checkpoint_path):
    model = Direct3DPoseNet(
        num_joints=cfg.num_joints,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
    ).to(cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


# -------------------------------
# Rendering 3D -> immagine 2D
# -------------------------------
def render_pose_3d(joints, bones, size=256):
    """Rende un'immagine 2D con la proiezione 3D della posa."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # Normalizza per rientrare nell'immagine
    joints_2d = joints.copy()
    joints_2d -= joints_2d.mean(axis=0)  # centra
    scale = np.max(np.abs(joints_2d)) + 1e-6
    joints_2d = joints_2d / scale * (size // 3) + size // 2

    # Disegna ossa
    for i, j in bones:
        p1 = tuple(joints_2d[i, :2].astype(int))
        p2 = tuple(joints_2d[j, :2].astype(int))
        cv2.line(img, p1, p2, (0, 0, 255), 2)

    # Disegna joints
    for (x, y, _) in joints_2d:
        cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)

    return img


# -------------------------------
# Inference su video
# -------------------------------
def run_inference(video_path, checkpoint_path):
    cfg = Config()
    set_seed(cfg.seed)

    model = load_model(cfg, checkpoint_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore apertura video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess: resize + to tensor
        img = cv2.resize(frame, (cfg.img_size, cfg.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(cfg.device)

        # dummy batch temporale (T=1 per demo)
        feat = torch.flatten(img, 1).unsqueeze(1)  # (B,1,D)
        with torch.no_grad():
            pred = model(feat)  # (B,J,3)
        joints = pred[0].cpu().numpy()

        # render ricostruzione 3D
        pose_img = render_pose_3d(joints, cfg.bones, size=frame.shape[0])

        # ridimensiona video a stessa altezza
        frame_resized = cv2.resize(frame, (pose_img.shape[1], pose_img.shape[0]))

        # affianca input e output
        out = cv2.hconcat([frame_resized, pose_img])
        cv2.imshow("Input | Ricostruzione 3D", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference("sample.mp4", "checkpoints/best.pth")
