import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def soft_argmax_coords(heatmaps, width=640, height=352):
    B_flat, C, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype
    
    # 1. Softmax spaziale
    hm_flat = heatmaps.view(B_flat, -1)
    hm_softmax = F.softmax(hm_flat, dim=-1).view(B_flat, 1, H, W)

    # 2. Creazione griglie di coordinate (Tua normalizzazione)
    pos_x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    y_range = height / width
    pos_y = torch.linspace(-y_range, y_range, H, device=device, dtype=dtype)
    
    # 3. Calcolo valore atteso
    expected_x = torch.sum(hm_softmax.sum(dim=2) * pos_x, dim=-1)
    expected_y = torch.sum(hm_softmax.sum(dim=3) * pos_y, dim=-1)
    
    return torch.stack([expected_x, expected_y], dim=-1), hm_softmax

def create_debug_video():
    path = "/mnt/proj2/eu-25-52/ar17/DroneDataset/train/video_00715/heatmaps.pt"
    if not os.path.exists(path):
        print(f"Errore: File non trovato in {path}")
        return

    # Caricamento heatmap: assumiamo [T, J, H, W]
    data = torch.load(path).cpu().float()
    if data.dim() != 4:
        print(f"Errore: Attesi 4 parametri [T, J, H, W], trovati {data.dim()}")
        return

    T, J, H, W = data.shape
    print(f"Sequenza caricata: T={T}, J={J}, H={H}, W={W}")

    # Setup della figura
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lim = 352 / 640
    
    # Funzione di aggiornamento per l'animazione
    def update(frame):
        ax.clear()
        
        # Estraiamo heatmap del frame corrente
        hms_frame = data[frame:frame+1] # [1, J, H, W]
        
        # Calcoliamo coordinate e softmax
        coords, hms_softmax = soft_argmax_coords(hms_frame.view(-1, 1, H, W))
        coords = coords.view(J, 2).numpy()
        
        # Somma delle heatmap per lo sfondo
        combined_hm = hms_softmax.sum(dim=0).squeeze().numpy()
        
        # Plot sfondo heatmap
        ax.imshow(combined_hm, extent=[-1, 1, y_lim, -y_lim], alpha=0.7, cmap='magma')
        
        # Plot keypoints estratti
        ax.scatter(coords[:, 0], coords[:, 1], c='cyan', marker='o', s=20, edgecolors='white', label='Soft-Argmax')
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(y_lim + 0.1, -y_lim - 0.1)
        ax.set_title(f"Frame {frame}/{T} - Video 00715")
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper right')

    # Creazione animazione
    ani = animation.FuncAnimation(fig, update, frames=range(0, T, 2), interval=50) # step 2 per velocità

    output_video = "debug_keypoints_video.mp4"
    print("Generazione video in corso...")
    
    # Richiede 'ffmpeg' installato nel sistema
    try:
        ani.save(output_video, writer='ffmpeg', fps=15)
        print(f"Video salvato con successo: {output_video}")
    except Exception as e:
        print(f"Errore durante il salvataggio del video: {e}")
        print("Assicurati di avere ffmpeg installato. Provo a salvare in HTML (JSAnim)...")
        ani.save("debug_keypoints.html", writer='html')

if __name__ == "__main__":
    create_debug_video()