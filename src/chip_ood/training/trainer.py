import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import random
import numpy as np
from tqdm import tqdm

from chip_ood.data import get_colored_mnist_loaders, DomainBedWrapper
from chip_ood.models import MNISTEncoder, ResNetEncoder, MNISTDecoder, ResNetDecoder, Predictor, AdversaryClf
from chip_ood.methods import ERM, CHIP, EIIL

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data
    if cfg.data.dataset_type == "mnist":
        train_loader, test_loader = get_colored_mnist_loaders(
            root=cfg.data.root,
            batch_size=cfg.data.batch_size,
            train_correlation=cfg.data.train_correlation,
            test_correlation=cfg.data.test_correlation,
            seed=cfg.seed,
            num_workers=cfg.data.num_workers
        )
        val_loader = test_loader # Use test as val for simple CMNIST
        
        # Define architecture params for MNIST
        enc_cls = MNISTEncoder
        dec_cls = MNISTDecoder
        input_dim_flat = 1 # Not used by these encoders
        
    elif cfg.data.dataset_type == "domainbed":
        db = DomainBedWrapper(
            root=cfg.data.root,
            dataset_name=cfg.data.dataset_name,
            test_envs=cfg.data.test_envs,
            seed=cfg.seed
        )
        # For simple splitting, get_loaders returns (train, [val_loaders...], [test_loaders...])
        train_loader, val_loaders, test_loaders = db.get_loaders(
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers
        )
        # Use first test loader for simplified loop
        test_loader = test_loaders[0]
        val_loader = val_loaders[0] if val_loaders else test_loader
        
        enc_cls = ResNetEncoder
        dec_cls = ResNetDecoder
    else:
        raise ValueError(f"Unknown dataset type {cfg.data.dataset_type}")

    # 2. Model components
    if "chip" in cfg.method.name:
        encoder = enc_cls(z_dim_c=cfg.method.z_dim_c, z_dim_s=cfg.method.z_dim_s).to(device)
        predictor = Predictor(input_dim=cfg.method.z_dim_c, num_classes=cfg.data.num_classes, hidden_dim=cfg.method.hidden_dim).to(device)
        decoder = dec_cls(z_dim=cfg.method.z_dim_c + cfg.method.z_dim_s).to(device)
        adversary = AdversaryClf(input_dim=cfg.method.z_dim_s, num_classes=cfg.data.num_classes).to(device)
        
        model = CHIP(encoder, predictor, decoder, adversary, 
                     lambda_rec=cfg.method.lambda_rec, 
                     lambda_adv=cfg.method.lambda_adv,
                     adv_type=cfg.method.adv_type).to(device)
                     
    elif cfg.method.name == "erm":
        encoder = enc_cls(z_dim_c=cfg.method.z_dim_c, z_dim_s=0).to(device)
        predictor = Predictor(input_dim=cfg.method.z_dim_c, num_classes=cfg.data.num_classes, hidden_dim=cfg.method.hidden_dim).to(device)
        model = ERM(encoder, predictor).to(device)
        
    elif cfg.method.name == "coral":
        encoder = enc_cls(z_dim_c=cfg.method.z_dim_c, z_dim_s=0).to(device)
        predictor = Predictor(input_dim=cfg.method.z_dim_c, num_classes=cfg.data.num_classes, hidden_dim=cfg.method.hidden_dim).to(device)
        model = CORAL(encoder, predictor).to(device)
        
    elif cfg.method.name == "irm":
        encoder = enc_cls(z_dim_c=cfg.method.z_dim_c, z_dim_s=0).to(device)
        predictor = Predictor(input_dim=cfg.method.z_dim_c, num_classes=cfg.data.num_classes, hidden_dim=cfg.method.hidden_dim).to(device)
        model = IRM(encoder, predictor).to(device)
        
    elif cfg.method.name == "groupdro":
        encoder = enc_cls(z_dim_c=cfg.method.z_dim_c, z_dim_s=0).to(device)
        predictor = Predictor(input_dim=cfg.method.z_dim_c, num_classes=cfg.data.num_classes, hidden_dim=cfg.method.hidden_dim).to(device)
        # Infer num_groups from data? Usually len(dataset.domains). 
        # For simplicity, pass strict num_groups via config or infer from data wrapper if exposed.
        # Let's assume 4 for PACS/VLCS for now or allow config to set it.
        num_groups = 4 # Default for PACS
        if "num_groups" in cfg.data: num_groups = cfg.data.num_groups
        model = GroupDRO(encoder, predictor, num_groups=num_groups).to(device)
        
    elif cfg.method.name == "eiil":
        encoder = enc_cls(z_dim_c=cfg.method.z_dim_c, z_dim_s=0).to(device)
        predictor = Predictor(input_dim=cfg.method.z_dim_c, num_classes=cfg.data.num_classes).to(device)
        model = EIIL(encoder, predictor, num_classes=cfg.data.num_classes).to(device)

    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
    
    # 4. Loop
    metrics_file = os.path.join(os.getcwd(), "metrics.jsonl")
    best_acc = 0.0
    
    # Schedule Params
    # Warmup ERM for 5 epochs
    warmup_epochs = 5
    # Then ramp up parameters
    
    for epoch in range(1, cfg.trainer.max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        adv_acc_sum = 0.0
        rec_loss_sum = 0.0
        steps = 0
        
        # Adjust lambdas based on schedule
        if "chip" in cfg.method.name:
            if epoch <= warmup_epochs:
                # ERM Phase
                model.lambda_rec = 0.0
                model.lambda_adv = 0.0
            else:
                # Full Phase (sudden on for now, or ramp?)
                # Stick to simple switch for now
                model.lambda_rec = cfg.method.lambda_rec
                model.lambda_adv = cfg.method.lambda_adv
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in train_loader:
            x, y, d = batch # Upgrade: expect triplets
            x, y, d = x.to(device), y.to(device), d.to(device)
            
            optimizer.zero_grad()
            
            # Method Update
            if hasattr(model, "update_with_extras"):
                loss, metrics = model.update_with_extras(x, y, {"d": d})
            else:
                # Legacy / Generic fallback
                loss, metrics = model.update(x, y, step=epoch)
            
            loss.backward()
            optimizer.step()
            
            rec_loss_sum += batch_metrics.get("loss_rec", 0.0)
            steps += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "acc": f"{batch_metrics.get('acc',0):.3f}",
                "adv": f"{batch_metrics.get('adv_acc',0):.2f}"
            })

        train_loss /= steps
        train_acc /= steps
        adv_acc_mean = adv_acc_sum / steps
        rec_loss_mean = rec_loss_sum / steps
        
        # Validation
        model.eval()
        test_acc = 0.0
        test_steps = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model.predict(x)
                acc = (logits.argmax(1) == y).float().mean()
                test_acc += acc.item()
                test_steps += 1
        test_acc /= test_steps
        
        # Log
        log_entry = {
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_acc": train_acc, 
            "test_acc": test_acc,
            "adv_acc": adv_acc_mean,
            "rec_loss": rec_loss_mean,
            "lambda_adv": getattr(model, "lambda_adv", 0.0)
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        print(f"Epoch {epoch}: Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, Adv Acc: {adv_acc_mean:.3f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pt")

    # Final summary
    with open("final_metrics.json", "w") as f:
        json.dump({
            "final_test_acc": test_acc, 
            "final_train_acc": train_acc,
            "best_test_acc": best_acc,
            "final_adv_acc": adv_acc_mean
        }, f, indent=2)

if __name__ == "__main__":
    main()
