import torch

# Carica i due mondi
vt = torch.load("debug_hm_vitpose_raw.pt") # Probabilmente (17, H, W)
us = torch.load("debug_hm_ours_raw.pt")    # Probabilmente (17, H, W)
var = torch.load("debug_logvar_ours_raw.pt") # (17,)

print(f"{'Joint':<10} | {'ViT Max':<10} | {'Ours Max':<10} | {'Ours LogVar':<12}")
print("-" * 50)

for j in range(us.shape[0]):
    v_max = vt[j].max().item()
    u_max = us[j].max().item()
    l_var = var[j].item()
    print(f"{j:<10} | {v_max:<10.4f} | {u_max:<10.4f} | {l_var:<12.4f}")