# imports
import torch
import matplotlib.pyplot as plt
from sportspose.dataset import SportsPoseDataset
from sportspose.plot_skeleton_3d import plot_skeleton_3d
from torch.utils.data import DataLoader
from torchvision import transforms
import os

datapath = "/DATA/SportsPose"

# Example of picking sequences based off requirements
dataset_jump_indoor = SportsPoseDataset(
    data_dir=datapath,
    sample_level="video",
    seq_size=10,  # 1.5 secondi di video
    whitelist={
        "metadata": 
            {"tag":"indoors",
             "activity": "jump"},
    }
)
#dataloader = torch.utils.data.DataLoader(dataset_jump_indoor, batch_size=2, shuffle=True)  

# Print basic dataset info
print(f"Dataset size: {len(dataset_jump_indoor)}")

# Inspect first sample structure
sample = dataset_jump_indoor[2]
print("\nKeys in first sample:")
for key in sample.keys():
    print(f"- {key}")
    if isinstance(sample[key], dict):
        for subkey in sample[key].keys():
            print(f"  └── {subkey}")
            if isinstance(sample[key][subkey], dict):
                for subsubkey in sample[key][subkey].keys():
                    print(f"      └── {subsubkey}")

# Print shapes of different components for first sample
print("\nShapes of components in first sample:")
print(f"Left image sequence: {sample['video']['image']['left'].shape}")
print(f"Right image sequence: {sample['video']['image']['right'].shape}")

# Get all video sequences from different views
views = ['front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left']

# Create dictionary to store all sequences
all_views = {}
for view in views:
    all_views[view] = sample["video"]["image"][view]
    print(f"{view} sequence shape: {all_views[view].shape}")
    print("Test", all_views[view].shape[0])

# Visualize first frame from all views
#rows, cols = 3, 3  # 3x3 grid (7 views + 2 empty spaces)
#plt.figure(figsize=(15, 15))
#
#for idx, view in enumerate(views):
#    plt.subplot(rows, cols, idx + 1)
#    plt.imshow(all_views[view][0])  # [0] selects first frame
#    plt.title(f"View: {view}")
#    plt.axis('off')
#
#plt.tight_layout()
#plt.savefig('all_views_first_frame.png')
#plt.close()

# sample contiene tipicamente:
#   sample["video"]: tensor (num_cams, T, H, W, 3)
#   sample["pose3d"]: tensor (T, J, 3)

videos = sample["video"]        # (7, T, H, W, 3)
pose3d = sample["joints_3d"]["data_points"]       # (T, J, 3)

os.makedirs("outputs", exist_ok=True)

# === 2. Cicla sulle 7 camere ===
for view in views:
    vid = videos["image"][view]
    frame0 = vid[0]  # primo frame

    fig = plt.figure(figsize=(10, 5))

    # Frame a sinistra
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(frame0)
    ax1.axis("off")
    ax1.set_title(f"Camera {view} - Frame 0")

    # Skeleton 3D a destra (usa la funzione del repo)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    pose3d_fixed = pose3d.copy()
    pose3d_fixed[..., 2] *= -1  # flip lungo Z
    plot_skeleton_3d(pose3d_fixed[0], ax=ax2, flip_z=False)
    ax2.set_title("3D Skeleton (GT)")

    out_path = f"outputs/sample_cam{view}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

print("Fatto! Immagini salvate in outputs/")