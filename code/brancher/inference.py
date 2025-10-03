import torch
import cv2
import numpy as np
from config import Config
from model import create_model
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sportspose.plot_skeleton_3d import plot_skeleton_3d


# -------------------------------
# Carica modello e pesi
# -------------------------------
def load_model(cfg, checkpoint_path):
    model = create_model()
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)


    # Se i pesi hanno il prefisso "module.", rimuovilo
    state_dict = ckpt["model_pos"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    # Carica SOLO dentro la parte DSTformer del Brancher
    model.dst.load_state_dict(new_state_dict, strict=False)

    model.eval()
    print(f"Loaded DSTformer weights from {checkpoint_path}")
    return model


def joints3d_to_img(joints3d, size):
    """
    Crea immagine con scheletro 3D usando matplotlib e ritorna np.uint8 (size,size,3).
    """
    fig = plt.figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)

    # subplot unico, con flip su Z (come nell'esperimento)
    ax = fig.add_subplot(111, projection="3d")

    # copia e flip Z
    pose3d_fixed = joints3d.copy()
    pose3d_fixed[..., 2] *= -1

    # plot scheletro
    plot_skeleton_3d(pose3d_fixed, ax=ax, flip_z=False)

    # limiti uguali su x,y,z (opzionale, rende proporzioni corrette)
    max_range = (pose3d_fixed.max() - pose3d_fixed.min()) / 2
    mid = pose3d_fixed.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.axis("off")
    plt.tight_layout()

    # render in numpy
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = img[..., :3]
    plt.close(fig)

    # resize quadrato
    img = cv2.resize(img, (size, size))
    return img


# -------------------------------
# Inference su video
# -------------------------------
def run_inference(video_path, checkpoint_path, out_path="output.avi"):
    cfg = Config()

    model = load_model(cfg, checkpoint_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore apertura video {video_path}")
        return

    # Prendi info dal video di input
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Inizializza VideoWriter (usa lo stesso fps, altezza e larghezza raddoppiata per affiancare input/output)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # puoi usare anche "mp4v" se vuoi .mp4
    out = cv2.VideoWriter(out_path, fourcc, fps, (width * 2, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess: resize + to tensor
        img = cv2.resize(frame, cfg.img_size)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(cfg.device)  # (1,1,H,W,3)

        with torch.no_grad(): 
            pred = model(img)  # (B,T,J,3)
        joints = pred[0, 0].cpu().numpy()  # (17,3)

        # crea immagine scheletro 3D
        pose_img = joints3d_to_img(joints, size=frame.shape[0])

        # ridimensiona pose_img a stessa altezza del frame originale
        pose_img_resized = cv2.resize(pose_img, (frame.shape[1], frame.shape[0]))

        # affianca input e output
        combined = cv2.hconcat([frame, pose_img_resized])

        # scrivi nel video
        out.write(combined)

    cap.release()
    out.release()
    print(f"Video salvato in: {out_path}")


if __name__ == "__main__":
    weights_path = "/Experiments/angeloa00/motionbert/best_epoch.bin"  # Percorso ai pesi salvati
    video_path = "AR17/code/video/IMG_3990.mp4"  # Percorso al video di input
    #video_path = "/DATA/SportsPose/videos/indoors/S00/Video_14_10_2022_103104/CAM0.avi"
    run_inference(video_path, weights_path)
