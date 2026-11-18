import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from fusion_dataloader import FusionDataset, CLASSES
from fusion_model import FusionThreatModel

# ================= CONFIGURATION =================
SAVE_DIR = r"D:\Threat_Detection_Workspace\03_Base_Model"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "base_threat_model_29_classes.pth")

EPOCHS = 35 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4 
# =================================================

def train_base_model():
    print("🚀 INITIALIZING TWO-HEADED MULTI-TASK TRAINING...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Hardware: {device.type.upper()}")

    print("⏳ Loading Datasets...")
    train_dataset = FusionDataset(split="train")
    test_dataset = FusionDataset(split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("⚖️ Calculating Binary & Fine-Grained Weights...")
    # 1. Fine-Grained Weights (29 Classes)
    class_counts = torch.zeros(len(CLASSES))
    normal_count, threat_count = 0, 0
    
    for _, _, _, label in train_dataset.samples:
        class_counts[label] += 1
        if label == 0: normal_count += 1
        else: threat_count += 1
        
    fine_weights = 1.0 / (class_counts + 1e-6)
    fine_weights = fine_weights / fine_weights.sum() * len(CLASSES)
    fine_weights = fine_weights.to(device)

    # 2. Binary Weights (Normal vs Threat)
    binary_weights = torch.tensor([1.0 / normal_count, 1.0 / threat_count])
    binary_weights = binary_weights / binary_weights.sum() * 2.0
    binary_weights = binary_weights.to(device)

    model = FusionThreatModel(num_classes=len(CLASSES)).to(device)
    
    # We now need TWO loss functions
    criterion_binary = nn.CrossEntropyLoss(weight=binary_weights)
    criterion_fine = nn.CrossEntropyLoss(weight=fine_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    print("\n" + "="*60)
    print("🧠 STARTING MULTI-TASK TRAINING LOOP")
    print("="*60)

    best_binary_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        model.train()
        total_loss, bin_correct, fine_correct, total = 0.0, 0, 0, 0
        
        train_loop = tqdm(train_loader, desc="Training", leave=False)
        for skeletons, spatial, labels in train_loop:
            skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)
            
            # Create Binary Labels (0 if Normal, 1 if any Threat)
            binary_labels = (labels > 0).long().to(device)

            optimizer.zero_grad()
            out_binary, out_fine = model(skeletons, spatial)
            
            # Calculate Both Losses
            loss_bin = criterion_binary(out_binary, binary_labels)
            loss_fine = criterion_fine(out_fine, labels)
            
            # 🌟 MULTI-TASK LEARNING MAGIC: Force the AI to prioritize Threat Detection (x2.0 weight)
            loss = (2.0 * loss_bin) + (1.0 * loss_fine)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)
            
            _, bin_pred = torch.max(out_binary.data, 1)
            bin_correct += (bin_pred == binary_labels).sum().item()
            
            _, fine_pred = torch.max(out_fine.data, 1)
            fine_correct += (fine_pred == labels).sum().item()
            
            train_loop.set_postfix(loss=loss.item())

        train_bin_acc = 100 * bin_correct / total
        train_fine_acc = 100 * fine_correct / total

        # Validation Phase
        model.eval()
        val_bin_correct, val_fine_correct, val_total = 0, 0, 0
        
        val_loop = tqdm(test_loader, desc="Validating", leave=False)
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
        current_lr = optimizer.param_groups[0]['lr']

        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f}")
        print(f"   🛡️ THREAT DETECTION (Binary) -> Train: {train_bin_acc:.2f}% | Val: {val_bin_acc:.2f}%")
        print(f"   🎯 THREAT TYPE (Fine)      -> Train: {train_fine_acc:.2f}% | Val: {val_fine_acc:.2f}%")

        # We schedule learning rate based on the crucial Binary accuracy
        scheduler.step(val_bin_acc)

        if val_bin_acc > best_binary_acc:
            best_binary_acc = val_bin_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   🌟 New Best Watchdog Model Saved! (Threat Detection Acc: {best_binary_acc:.2f}%)")

if __name__ == "__main__":
    train_base_model()