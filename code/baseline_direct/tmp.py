# imports
import torch
import matplotlib.pyplot as plt
from sportspose.dataset import SportsPoseDataset
from sportspose.plot_skeleton_3d import plot_skeleton_3d
from torch.utils.data import DataLoader
from torchvision import transforms

datapath = "/DATA/SportsPose"

# Example of picking sequences based off requirements
dataset_jump_indoor = SportsPoseDataset(
    data_dir=datapath,
    sample_level="video",
    seq_size=135,  # 1.5 secondi di video
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
sample = dataset_jump_indoor[0]
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
if 'pose3d' in sample:
    print(f"3D pose shape: {sample['pose3d'].shape}")
if 'pose2d' in sample:
    print(f"2D pose shape: {sample['pose2d'].shape}")

# Get first sample
sample = dataset_jump_indoor[0]

# Get all video sequences from different views
views = ['front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left']

# Create dictionary to store all sequences
all_views = {}
for view in views:
    all_views[view] = sample["video"]["image"][view]
    print(f"{view} sequence shape: {all_views[view].shape}")

# Visualize first frame from all views
rows, cols = 3, 3  # 3x3 grid (7 views + 2 empty spaces)
plt.figure(figsize=(15, 15))

for idx, view in enumerate(views):
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(all_views[view][0])  # [0] selects first frame
    plt.title(f"View: {view}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('all_views_first_frame.png')
plt.close()