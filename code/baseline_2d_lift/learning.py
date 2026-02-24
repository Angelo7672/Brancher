import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

# -------------------------------
# Carica modello e pesi
# -------------------------------
def load_weights(model, checkpoint_path, device):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len("module."):]
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded {len(new_state_dict)} layers | Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    return model
