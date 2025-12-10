import os
import json
import torch

v3_dir = "/Experiments/angeloa00/AirPose_v2/train"

# === Dimensioni del video (fisse per SportsPose) ===
W_VIDEO = 416
H_VIDEO = 640


# === Estrazione keypoints da UNA heatmap ===
def extract_keypoints_from_heatmap(hmap):
    """
    hmap: [J, Hh, Wh]
    returns: [J, 3] -> (x, y, conf) in VIDEO coordinates
    """
    J, Hh, Wh = hmap.shape
    keypoints = torch.zeros((J, 3), dtype=torch.float32)

    for j in range(J):
        h = hmap[j]  # [Hh, Wh]
        conf, idx = torch.max(h.view(-1), 0)
        y_h, x_h = divmod(int(idx.item()), Wh)  # coords in heatmap space

        # === Riscalatura alla risoluzione video ===
        x_v = int(x_h / Wh * W_VIDEO)
        y_v = int(y_h / Hh * H_VIDEO)

        keypoints[j] = torch.tensor([x_v, y_v, conf])

    return keypoints


# === Iterazione su tutte le cartelle ===
for sample_folder in sorted(os.listdir(v3_dir)):
    folder_path = os.path.join(v3_dir, sample_folder)
    if not os.path.isdir(folder_path):
        continue

    heatmap_path = os.path.join(folder_path, "heatmaps.pt")
    if not os.path.isfile(heatmap_path):
        print(f"Heatmap missing in {sample_folder}, skipping.")
        continue

    print(f"Processing {sample_folder}")

    # Carica heatmap: [T, J, Hh, Wh]
    heatmaps = torch.load(heatmap_path).float()
    T, J, Hh, Wh = heatmaps.shape
    print(f"  Loaded heatmaps: {heatmaps.shape}")

    all_keypoints = []   # lista lunga T, ognuna [J,3]

    # === Ciclo su tutti i frame ===
    for t in range(T):
        kpts = extract_keypoints_from_heatmap(heatmaps[t])  # [J,3]
        all_keypoints.append(kpts.tolist())

    # === Salvataggio JSON ===
    save_path = os.path.join(folder_path, "pose2d.json")

    data = {
        "num_frames": T,
        "num_joints": J,
        "keypoints": all_keypoints
    }

    with open(save_path, "w") as f:
        json.dump(data, f)

    print(f"  Saved 2D keypoints → {save_path}")

print("All folders processed.")
