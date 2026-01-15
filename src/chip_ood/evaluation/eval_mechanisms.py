import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from chip_ood.data.colored_mnist import ColoredMNIST
from chip_ood.data.domainbed_wrapper import DomainBedWrapper
from chip_ood.models import MNISTEncoder, MNISTDecoder, Predictor, AdversaryClf, ResNetEncoder, ResNetDecoder
from chip_ood.methods.chip import CHIP
from omegaconf import OmegaConf

def load_chip_model(run_dir, device):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    
    # 1. Dataset & Architecture
    if cfg.data.dataset_type == "colored_mnist":
        enc_cls = MNISTEncoder
        dec_cls = MNISTDecoder
    else:
        enc_cls = ResNetEncoder
        dec_cls = ResNetDecoder
        
    # 2. Rebuild Model (Architecture only)
    # Note: We need to match exactly how trainer.py builds it
    z_dim_c = cfg.method.z_dim_c
    z_dim_s = cfg.method.z_dim_s
    
    if "chip" in cfg.method.name:
        encoder = enc_cls(z_dim_c=z_dim_c, z_dim_s=z_dim_s).to(device)
        predictor = Predictor(input_dim=z_dim_c, num_classes=cfg.data.num_classes, hidden_dim=cfg.method.hidden_dim).to(device)
        decoder = dec_cls(z_dim=z_dim_c + z_dim_s).to(device)
        adversary = AdversaryClf(input_dim=z_dim_s, num_classes=cfg.data.num_classes).to(device)
        
        model = CHIP(encoder, predictor, decoder, adversary, 
                     lambda_rec=cfg.method.lambda_rec, 
                     lambda_adv=cfg.method.lambda_adv,
                     adv_type=cfg.method.adv_type).to(device)
    else:
        raise ValueError(f"eval_mechanisms only supports CHIP models for now, found {cfg.method.name}")

    # 3. Load Weights
    path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(path):
        # Try checkpoint_latest
        path = os.path.join(run_dir, "checkpoint_last.pt")
        
    try:
        state_dict = torch.load(path, map_location=device)
        # Trainer saves 'model' state dict wrapper? 
        # Usually torch.save(model.state_dict()) or {'model': ...}
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        model.load_state_dict(state_dict)
        print(f"Loaded model from {path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, cfg

    return model, cfg

def get_data(cfg, device):
    # Load test data
    if cfg.data.dataset_type == "colored_mnist":
        # Load test env
        ds = ColoredMNIST(root=cfg.data.root, env="test", correlation=0.1, seed=0, variant="binary_irm")
        return ds, "cmnist"
    
    elif cfg.data.dataset_type == "domainbed":
        # Load a held-out domain or just a validator
        # For probing, we want a MIX of domains or a specific domain?
        # To probe "Invariant to Domain", we need Multiple Domains.
        # Let's load the Train Env (mixed) to see if we can predict domain from Zc.
        # Ideally, Zc should NOT predict domain.
        
        # We'll re-instantiate Wrapper
        db = DomainBedWrapper(cfg.data.dataset_name, cfg.data.root, list(cfg.data.test_envs), 
                              holdout_fraction=0.1, seed=0)
        
        # We need a loader that yields (x, y, d)
        # We can use the train_loader (pooled training domains)
        loader, _, _ = db.get_loaders(batch_size=64, num_workers=4)
        return loader, "domainbed"
        
    return None, "unknown"

def eval_mechanisms(run_dir, device="cpu"):
    print(f"\n--- Audit: {run_dir} ---")
    model, cfg = load_chip_model(run_dir, device)
    if model is None: return

    model.eval()
    data_obj, d_type = get_data(cfg, device)
    
    # Collect Representations
    z_c_list, z_s_list, y_list, d_list = [], [], [], []
    
    max_samples = 2000
    count = 0
    
    print("Extracting features...")
    with torch.no_grad():
        if d_type == "domainbed":
            for x, y, d in data_obj:
                x = x.to(device)
                zc, zs = model.encoder(x)
                
                z_c_list.append(zc.cpu().numpy())
                z_s_list.append(zs.cpu().numpy())
                y_list.append(y.cpu().numpy())
                d_list.append(d.cpu().numpy())
                
                count += x.size(0)
                if count >= max_samples: break
                
        elif d_type == "cmnist":
            for i in range(min(len(data_obj), max_samples)):
                img, y = data_obj[i]
                # Infer color/domain
                r = img[0].sum()
                g = img[1].sum()
                c = 0 if r > g else 1 # Treat 'color' as 'domain'
                
                img_t = img.unsqueeze(0).to(device)
                zc, zs = model.encoder(img_t)
                
                z_c_list.append(zc.cpu().numpy())
                z_s_list.append(zs.cpu().numpy())
                y_list.append(y.item())
                d_list.append(c)

    Zc = np.concatenate(z_c_list, axis=0) if z_c_list else np.array([])
    Zs = np.concatenate(z_s_list, axis=0) if z_s_list else np.array([])
    Y = np.concatenate(y_list, axis=0) if isinstance(y_list[0], np.ndarray) else np.array(y_list)
    D = np.concatenate(d_list, axis=0) if isinstance(d_list[0], np.ndarray) else np.array(d_list)
    
    if len(Zc) == 0:
        print("No data extracted.")
        return

    # Probing
    results = {}
    
    def probe(X, target, name):
        if len(np.unique(target)) < 2:
            return 0.0
        clf = LogisticRegression(max_iter=1000).fit(X, target)
        return clf.score(X, target)

    # Zc -> Y (Should be High)
    results["acc_zc_y"] = probe(Zc, Y, "Zc->Y")
    
    # Zc -> D (Should be Low for Invariance)
    results["acc_zc_d"] = probe(Zc, D, "Zc->D")
    
    # Zs -> Y (Should be Low for Disentanglement)
    results["acc_zs_y"] = probe(Zs, Y, "Zs->Y")
    
    # Zs -> D (Could be High if Zs captures domain info)
    results["acc_zs_d"] = probe(Zs, D, "Zs->D")
    
    print(json.dumps(results, indent=2))
    
    # Save
    out_path = os.path.join(run_dir, "diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        eval_mechanisms(sys.argv[1], device="mps" if torch.backends.mps.is_available() else "cpu")
