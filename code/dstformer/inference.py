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
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def joints3d_to_img(joints3d, size):
    """Crea immagine con scheletro 3D usando matplotlib e ritorna np.uint8 (size,size,3)."""
    fig = plt.figure(figsize=(4,4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")

    # plot skeleton
    plot_skeleton_3d(joints3d, ax=ax)

    # set limiti uguali su x,y,z
    max_range = (joints3d.max() - joints3d.min()) / 2
    mid = joints3d.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

    ax.axis("off")
    plt.tight_layout()

    # render fig in numpy
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    # resize a `size x size`
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
    weights_path = "/Experiments/angeloa00/dstformer/model_20250926-183734/weight_epoch9.pth"  # Percorso ai pesi salvati
    #video_path = "AR17/code/video/IMG_3990.mp4"  # Percorso al video di input
    video_path = "/DATA/SportsPose/videos/indoors/S00/Video_14_10_2022_103104/CAM0.avi"
    run_inference(video_path, weights_path)
