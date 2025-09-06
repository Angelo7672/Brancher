import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary
from config import Config

# -------------------------------
# Utilities
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding for sequences.
       senza positional encoding il Transformer vedrebbe solo un insieme di vettori indistinguibili per ordine. Con queste sinusoidi, ogni timestep diventa unico, e l'attenzione 
       può distinguere “frame 1, frame 2, frame 3…” e catturare la dinamica nel tempo.
    """
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # matrice con le codifiche
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # position = indici temporali da 0 a max_len-1, reshaped in (max_len, 1).
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # fattore di scala per le frequenze
        # Ogni posizione è rappresentata da una combinazione unica di onde.
        pe[:, 0::2] = torch.sin(position * div_term) # posizioni pari: seno
        pe[:, 1::2] = torch.cos(position * div_term) # posizioni dispari: coseno
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d). Aggiunge la matrice pe come buffer del modulo, non come parametro allenabile.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :] # Si prende solo la fetta di pe lunga T (cioè le prime T posizioni), poi si somma a x per aggiungere l'informazione posizionale.
        # Ora ogni frame ha incorporata l’informazione sulla sua posizione temporale.


# -------------------------------
# Model
# -------------------------------
class FrameEncoderResNet18(nn.Module):
    """ResNet-18 trunk that outputs a 512-d feature per frame.
    We apply it to (B, T, 3, H, W) by merging the B and T dims for speed.
    L'idea: applichiamo la ResNet a ciascun frame indipendentemente.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None) # load pretrained
        # Remove final fc layer and avgpool
        modules = list(base.children())[:-1]  # up to global avgpool
        self.backbone = nn.Sequential(*modules)  # outputs (N, 512, 1, 1)
        self.out_dim = 512 # feature dim frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W) input is a batch of T-frame clips
        # Merge B and T for efficiency → (B*T, 3, H, W). La ResNet-18 è pensata per immagini singole di forma (N, 3, H, W) (dove N = batch di immagini).
        # In altre parole: “dimentichiamo per un attimo” la dimensione temporale e trattiamo i frame come se fossero indipendenti.
        # L'appiattimento è un trucco per usare una CNN 2D su un input video, frame per frame, in un unico passo efficiente.
        B, T = x.shape[0], x.shape[1]
        x = x.view(B*T, *x.shape[2:]) # (B*T, 3, H, W)
        feats = self.backbone(x)  # (B*T, 512, 1, 1)
        feats = feats.view(B, T, self.out_dim) # (B, T, 512)
        return feats  # (B, T, 512). Ora abbiamo una sequenza di feature 512-d per ogni frame.


class TemporalTransformer(nn.Module):
    """Questo modulo prende la sequenza di feature estratte dai singoli frame e le trasforma in una rappresentazione che tiene conto delle relazioni temporali, così la rete non guarda 
       i frame in isolamento ma come parti di un movimento coerente.
    """
    def __init__(self, d_model: int, depth: int, nhead: int, mlp_ratio: float, drop: float):
        """d_model → dimensione delle feature in input (es. 512).
           depth → numero di strati Transformer.
           nhead → numero di teste di attenzione.
           mlp_ratio → fattore di espansione del feed-forward interno.
           drop → dropout per regolarizzazione.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),   # ampiezza della MLP interna
            dropout=drop,
            activation='gelu',
            batch_first=True,   # input shape (B, T, D)
            norm_first=True,    # LayerNorm prima di ogni sub-layer
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth) # stack di strati Transformer
        self.pos = PositionalEncoding(d_model)  # modulo che aggiunge positional encoding (indispensabile, perché il Transformer di base non conosce l’ordine dei frame).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.pos(x) # aggiunge informazioni sull’ordine temporale dei frame.
        x = self.encoder(x) # applica self-attention e feed-forward su tutta la sequenza per catturare relazioni a lungo raggio tra i frame.
        return x  # (B, T, 512) # stessa forma dell’input ma arricchito di contesto temporale.


class PoseHead(nn.Module):
    def __init__(self, d_model: int, num_joints: int, drop: float):
        """d_model = dimensione delle feature in input.
           num_joints = quanti giunti 3D vuoi stimare.
           drop = tasso di dropout per regolarizzazione.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), # normalizza ogni vettore feature.
            nn.Linear(d_model, d_model*2),  # proietta a uno spazio intermedio più ampio.
            nn.GELU(), # attivazione non lineare moderna, più smooth della ReLU.
            nn.Dropout(drop),
            nn.Linear(d_model*2, num_joints*3), # output finale: per ogni frame produce num_joints × 3 numeri (coordinate 3D).
        )
        self.num_joints = num_joints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = self.mlp(x) # La MLP trasforma ogni vettore di dimensione D in num_joints*3.
        B, T = x.shape[:2]
        y = y.view(B, T, self.num_joints, 3) # reshape a (B, T, J, 3) per interpretare correttamente le coordinate 3D di ogni giunto.
        return y


class Direct3DPoseNet(nn.Module):
    def __init__(self, num_joints: int, d_model: int, depth: int, heads: int, mlp_ratio: float, drop: float):
        """num_joints → quanti giunti vogliamo predire.
           d_model → dimensione delle feature (qui deve essere 512, per coerenza con ResNet-18).
           depth, heads, mlp_ratio, drop → iperparametri del Transformer temporale.
        """
        super().__init__()
        self.encoder = FrameEncoderResNet18(pretrained=True)
        assert self.encoder.out_dim == d_model, "Set embed_dim=512 to match ResNet18." # Per evitare mismatch
        self.temporal = TemporalTransformer(d_model, depth, heads, mlp_ratio, drop)
        self.head = PoseHead(d_model, num_joints, drop)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, 3, H, W)
        feats = self.encoder(video)        # (B, T, 512)
        tokens = self.temporal(feats)      # (B, T, 512)
        pose3d = self.head(tokens)         # (B, T, J, 3)
        # Root-centering (optional): ensure pelvis ~ 0
        root = pose3d[:, :, :1, :]         # joint 0 assumed pelvis
        pose3d = pose3d - root  # Lo si porta a (0,0,0) per rendere le pose root-centered → serve per rendere le pose confrontabili senza dipendere dalla posizione assoluta nello spazio.S
        return pose3d

# -------------------------------
# Factory
# -------------------------------
def create_model():
    cfg = Config()
    set_seed(cfg.seed)

    # Model
    model = Direct3DPoseNet(
        num_joints=cfg.num_joints,
        d_model=cfg.embed_dim,
        depth=cfg.transf_depth,
        heads=cfg.transf_heads,
        mlp_ratio=cfg.transf_mlp_ratio,
        drop=cfg.drop,
    ).to(cfg.device)

    print(f"Model params: {count_params(model)/1e6:.2f}M")

    # Schema del modello
    # Input fittizio: (B, T, 3, H, W) -> batch_size=1
    summary(
        model,
        input_size=(1, cfg.clip_len, 3, cfg.img_size, cfg.img_size),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        device=cfg.device,
    )

    return model