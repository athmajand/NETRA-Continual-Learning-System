import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from fusion_dataloader import FusionDataset, CLASSES
from fusion_model import FusionThreatModel

# ================= CONFIGURATION =================
SAVE_DIR = r"D:\Threat_Detection_Workspace\03_Base_Model"
BASE_MODEL_PATH = os.path.join(SAVE_DIR, "base_threat_model_29_classes.pth")
NEW_MODEL_PATH = os.path.join(SAVE_DIR, "ewc_continual_model_31_classes.pth")

EPOCHS = 15 
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EWC_LAMBDA = 2000  # The mathematical anchor protecting old memories
# =================================================

def train_ewc():
    print("🚀 INITIATING EWC CONTINUAL LEARNING SURGERY...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("⏳ Loading Datasets...")
    train_dataset = FusionDataset(split="train")
    test_dataset = FusionDataset(split="test")
    
    # --- DATASET SEGREGATION ---
    print("🧬 Segregating New Memories and Replay Buffer...")
    new_samples = [s for s in train_dataset.samples if s[3] >= 29]
    old_samples = [s for s in train_dataset.samples if s[3] < 29]
    
    # We take 1000 random old samples to calculate the Fisher Matrix & keep the Watchdog grounded
    random.shuffle(old_samples)
    fisher_buffer = old_samples[:1000]
    
    # The AI will only train on the New Data + a tiny memory buffer
    train_dataset.samples = new_samples + fisher_buffer
    random.shuffle(train_dataset.samples)
    
    fisher_dataset = copy.deepcopy(train_dataset)
    fisher_dataset.samples = fisher_buffer
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    fisher_loader = DataLoader(fisher_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 1. LOAD OLD BRAIN ---
    print("🧠 Loading Original 29-Class Brain...")
    model = FusionThreatModel(num_classes=29).to(device)
    model.load_state_dict(torch.load(BASE_MODEL_PATH))
    criterion = nn.CrossEntropyLoss()

    # --- 2. CALCULATE FISHER INFORMATION MATRIX ---
    print("🧮 Calculating Fisher Matrix (Mapping core memories)...")
    fisher_matrix = {}
    old_params = {}
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            fisher_matrix[n] = torch.zeros_like(p.data)
            old_params[n] = p.data.clone()

    model.eval()
    for skeletons, spatial, labels in tqdm(fisher_loader, desc="Fisher", leave=False):
        skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)
        model.zero_grad()
        _, out_fine = model(skeletons, spatial)
        loss = criterion(out_fine, labels)
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher_matrix[n].data += p.grad.data ** 2 / len(fisher_loader)

    # --- 3. NEURAL NETWORK SURGERY ---
    print("🔪 Performing Surgery: Expanding cortex to 31 classes...")
    old_weight = model.fine_head[4].weight.data.clone()
    old_bias = model.fine_head[4].bias.data.clone()
    
    # Grafting the new layer
    model.fine_head[4] = nn.Linear(128, 31).to(device)
    model.fine_head[4].weight.data[:29] = old_weight
    model.fine_head[4].bias.data[:29] = old_bias

    # --- 4. CONTINUAL TRAINING ---
    print("\n" + "="*60)
    print("🧠 STARTING EWC ASSIMILATION TRAINING")
    print("="*60)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_bin_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        model.train()
        total_loss, bin_correct, total = 0.0, 0, 0
        
        train_loop = tqdm(train_loader, desc="EWC Training", leave=False)
        for skeletons, spatial, labels in train_loop:
            skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)
            binary_labels = (labels > 0).long().to(device)

            optimizer.zero_grad()
            out_binary, out_fine = model(skeletons, spatial)
            
            loss_bin = criterion(out_binary, binary_labels)
            loss_fine = criterion(out_fine, labels)
            loss_task = (2.0 * loss_bin) + (1.0 * loss_fine)
            
            # 🛡️ THE EWC PENALTY (Protecting Old Memories)
            loss_ewc = 0
            for name, param in model.named_parameters():
                if name in fisher_matrix and name in old_params:
                    fisher = fisher_matrix[name].to(device)
                    old_val = old_params[name].to(device)
                    
                    # 🌟 THE SURGERY FIX: If the layer was expanded, only penalize the old neurons
                    if param.shape != old_val.shape:
                        current_param = param[:old_val.shape[0]] 
                    else:
                        current_param = param
                        
                    loss_ewc += (fisher * (current_param - old_val) ** 2).sum()
                    
            loss = loss_task + (EWC_LAMBDA / 2) * loss_ewc
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)
            _, bin_pred = torch.max(out_binary.data, 1)
            bin_correct += (bin_pred == binary_labels).sum().item()
            train_loop.set_postfix(loss=loss.item())

        train_bin_acc = 100 * bin_correct / total

        # VALIDATION ON ALL 31 CLASSES
        model.eval()
        val_bin_correct, val_fine_correct, val_total = 0, 0, 0
        
        val_loop = tqdm(test_loader, desc="Validating 31-Classes", leave=False)
        with torch.no_grad():
            for skeletons, spatial, labels in val_loop:
                skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)
                binary_labels = (labels > 0).long().to(device)

                out_binary, out_fine = model(skeletons, spatial)
                
                _, bin_pred = torch.max(out_binary.data, 1)
                val_bin_correct += (bin_pred == binary_labels).sum().item()
                
                _, fine_pred = torch.max(out_fine.data, 1)
                val_fine_correct += (fine_pred == labels).sum().item()
                val_total += labels.size(0)
                
        val_bin_acc = 100 * val_bin_correct / val_total
        val_fine_acc = 100 * val_fine_correct / val_total

        print(f"✅ Epoch [{epoch+1}/{EPOCHS}]")
        print(f"   🛡️ THREAT DETECTION (Binary) -> Train: {train_bin_acc:.2f}% | Val: {val_bin_acc:.2f}%")
        print(f"   🎯 THREAT TYPE (31 Classes)  -> Val: {val_fine_acc:.2f}%")

        if val_bin_acc > best_bin_acc:
            best_bin_acc = val_bin_acc
            torch.save(model.state_dict(), NEW_MODEL_PATH)
            print(f"   🌟 New 31-Class Continual Model Saved! (Watchdog Acc: {best_bin_acc:.2f}%)")

if __name__ == "__main__":
    train_ewc()