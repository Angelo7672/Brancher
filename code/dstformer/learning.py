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
def load_pretrained_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # 1. Pulizia prefissi comuni
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 2. Prepariamo il dizionario per il tuo modello
    model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in clean_dict.items():
        # Caso A: Il layer appartiene al backbone (MotionAGFormer)
        # Lo mappiamo sotto 'ag.'
        if f"ag.{k}" in model_dict:
            new_state_dict[f"ag.{k}"] = v
            
        # Caso B: Il layer è la testa finale (il tuo 'head' -> 'head3d.mlp.2')
        # Basandoci sul tuo debug, 'head' nel file deve andare in 'head3d.mlp.2'
        elif k == 'head.weight':
            new_state_dict['head3d.mlp.2.weight'] = v
        elif k == 'head.bias':
            new_state_dict['head3d.mlp.2.bias'] = v
            
        # Caso C: La parte intermedia (rep_logit / fc)
        elif k == 'rep_logit.fc.weight':
            new_state_dict['head3d.mlp.0.weight'] = v
        elif k == 'rep_logit.fc.bias':
            new_state_dict['head3d.mlp.0.bias'] = v

    # 3. Caricamento effettivo
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Caricamento completato!")
    print(f"   Matched: {len(new_state_dict)} layers")
    print(f"   Missing (dovrebbero essere solo extractor): {len(msg.missing_keys)}")
    
    return model
