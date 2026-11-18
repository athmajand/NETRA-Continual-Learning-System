import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion_model import FusionThreatModel
from fusion_dataloader import FusionDataset

# ================= CONFIGURATION =================
MODEL_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\base_threat_model_29_classes.pth"
SAVE_DIR = r"D:\Threat_Detection_Workspace\03_Base_Model"

EWC_LAMBDA = 5000  
EPOCHS = 2  
LEARNING_RATE = 0.001
# =================================================

def perform_neural_surgery(device):
    print("\n🧠 Loading Golden 31-Class Baseline (89.60% Accuracy)...")
    model = FusionThreatModel(num_classes=31).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    print("💾 Backing up established neural pathways...")
    old_weight = model.analyst_head.weight.data.clone()
    old_bias = model.analyst_head.bias.data.clone()
    
    print("✂️ Performing Neural Surgery: Expanding to 32 Classes...")
    model.analyst_head = nn.Linear(in_features=128, out_features=32).to(device)
    
    print("🧬 Grafting old memories into new architecture...")
    with torch.no_grad():
        model.analyst_head.weight.data[:31] = old_weight
        model.analyst_head.bias.data[:31] = old_bias

    return model

def compute_fisher_matrix(model, dataloader, device):
    print("\n🧮 Computing Fisher Information Matrix (Mapping Core Memories)...")
    fisher_dict = {}
    optpar_dict = {}
    
    for name, param in model.named_parameters():
        optpar_dict[name] = param.data.clone()
        fisher_dict[name] = torch.zeros_like(param.data)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    sample_count = 0
    for skeletons, spatial, labels in tqdm(dataloader, desc="Calculating Gradients"):
        skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)
        model.zero_grad()
        
        _, out_fine = model(skeletons, spatial)
        loss = criterion(out_fine, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        sample_count += 1
        if sample_count > 20: 
            break

    for name in fisher_dict:
        fisher_dict[name] /= sample_count

    return fisher_dict, optpar_dict

def train_ewc():
    print("🚀 INITIATING ULTIMATE EWC: IRON SHIELD + REHEARSAL + BATCHNORM LOCK...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    model = perform_neural_surgery(device)
    
    print("\n📂 Loading Data...")
    train_dataset = FusionDataset(split="train")
    fisher_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    fisher_dict, optpar_dict = compute_fisher_matrix(model, fisher_loader, device)

    print("\n🧊 Freezing the Core Brain (ResNet & LSTM)...")
    for name, param in model.named_parameters():
        if "analyst_head" not in name:
            param.requires_grad = False
            
    optimizer = optim.Adam(model.analyst_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("\n⚡ Commencing Iron Shield Assimilation Training...")
    
    # ==========================================
    # 🛡️ THE BATCHNORM FIX
    # ==========================================
    # We put the ENTIRE model into eval mode so the ResNet/LSTM BatchNorm 
    # statistics are permanently locked and cannot be corrupted.
    model.eval() 
    # We ONLY put the final linear layer into train mode.
    model.analyst_head.train()
    # ==========================================

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(fisher_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for skeletons, spatial, labels in pbar:
            skel_batch = skeletons.to(device)
            spat_batch = spatial.to(device)
            lbl_batch = labels.to(device)

            optimizer.zero_grad()
            _, out_fine = model(skel_batch, spat_batch)
            
            classification_loss = criterion(out_fine, lbl_batch)
            
            ewc_loss = 0
            for name, param in model.named_parameters():
                if name in fisher_dict and param.requires_grad: 
                    fisher_val = fisher_dict[name]
                    optpar_val = optpar_dict[name]
                    ewc_loss += (fisher_val * (param - optpar_val) ** 2).sum()
            
            loss = classification_loss + (EWC_LAMBDA / 2) * ewc_loss
            loss.backward()

            with torch.no_grad():
                if model.analyst_head.weight.grad is not None:
                    model.analyst_head.weight.grad[:31] = 0.0
                if model.analyst_head.bias.grad is not None:
                    model.analyst_head.bias.grad[:31] = 0.0

            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        epoch_save_path = os.path.join(SAVE_DIR, f"evolved_threat_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        tqdm.write(f"💾 CHECKPOINT SECURED: {epoch_save_path}")
            
    final_save_path = os.path.join(SAVE_DIR, "evolved_threat_model_32_classes_FINAL.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"\n✅ EVOLUTION COMPLETE. Final model saved to: {final_save_path}")

if __name__ == "__main__":
    train_ewc()