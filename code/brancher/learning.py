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
# Load Model and Weights
# -------------------------------
def load_pretrained_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in clean_dict.items():
        # Case A: The layer belongs to the backbone (MotionAGFormer)
        # We map it under 'ag.'
        if f"ag.{k}" in model_dict:
            new_state_dict[f"ag.{k}"] = v
            
        # Case B: The layer is the final head (your 'head' -> 'head3d.mlp.2')
        elif k == 'head.weight':
            new_state_dict['head3d.mlp.2.weight'] = v
        elif k == 'head.bias':
            new_state_dict['head3d.mlp.2.bias'] = v
            
        # Case C: The middle part (rep_logit / fc)
        elif k == 'rep_logit.fc.weight':
            new_state_dict['head3d.mlp.0.weight'] = v
        elif k == 'rep_logit.fc.bias':
            new_state_dict['head3d.mlp.0.bias'] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loading Completed!")
    print(f"   Matched: {len(new_state_dict)} layers")
    print(f"   Missing (they should only be extractors): {len(msg.missing_keys)}")
    
    return model


def load_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in clean_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                print(f"⚠️ Dimension mismatch for {k}: skip.")

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loading Completed!")
    print(f"   Matched: {len(new_state_dict)} layers | Missing: {len(msg.missing_keys)} | Unexpected: {len(msg.unexpected_keys)}")
    
    return model