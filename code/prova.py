import torch
import torch.nn as nn
from mmpose.apis import init_model

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # se hai una GPU


class ViTPoseBase(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=False):
        super(ViTPoseBase, self).__init__()
        
        # Carica il modello ViTPose pre-addestrato
        self.model = init_model(config_file, checkpoint_file, device=device)
        
        # Se vuoi congelare il backbone
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Salva le componenti principali
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        """
        x: Tensor di forma [B, 3, H, W]
        Output: heatmap [B, num_joints, H_out, W_out]
        """
        # Estrai feature dal backbone
        features = self.backbone(x)
        
        # Passa attraverso deconv + final layer (testa)
        x = self.deconv_layer(features)
        x = self.final_layer(x)
        return x

if __name__ == '__main__':

    # Sostituisci con i percorsi locali o URL corretti
    config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/vitpose/vitpose-b-multi-coco-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/vitpose/vitpose-b-multi-coco-256x192-608e5d8c_20230314.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViTPoseBase(config_file, checkpoint_file, device=device).to(device)
    model.eval()

    # Input dummy
    dummy_input = torch.randn(1, 3, 256, 192).to(device)
    output = model(dummy_input)

    print('Output shape:', output.shape)  # Tipicamente [1, 17, 64, 48]