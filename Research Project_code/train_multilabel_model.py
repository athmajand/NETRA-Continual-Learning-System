import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from fusion_dataloader import FusionDataset, CLASSES
from fusion_model import FusionThreatModel

# ================= CONFIGURATION =================
SAVE_DIR = r"D:\Threat_Detection_Workspace\03_Base_Model"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "multilabel_threat_model.pth")

EPOCHS = 35 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4 

# 🌟 NEW: The AI must be at least 80% confident to trigger a threat class!
CONFIDENCE_THRESHOLD = 0.80  
# =================================================

def train_multilabel_model():
    print(f"🚀 INITIALIZING STRICT MULTI-LABEL TRAINING (Threshold: {CONFIDENCE_THRESHOLD*100}%)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Hardware: {device.type.upper()}")

    print("⏳ Loading Datasets...")
    train_dataset = FusionDataset(split="train")
    test_dataset = FusionDataset(split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("⚖️ Calculating Binary & Fine-Grained Weights...")
    class_counts = torch.zeros(len(CLASSES))
    normal_count, threat_count = 0, 0
    
    for _, _, _, label in train_dataset.samples:
        class_counts[label] += 1
        if label == 0: normal_count += 1
        else: threat_count += 1
        
    fine_weights = 1.0 / (class_counts + 1e-6)
    fine_weights = fine_weights / fine_weights.sum() * len(CLASSES)
    fine_weights = fine_weights.to(device)

    binary_weights = torch.tensor([1.0 / normal_count, 1.0 / threat_count])
    binary_weights = binary_weights / binary_weights.sum() * 2.0
    binary_weights = binary_weights.to(device)

    model = FusionThreatModel(num_classes=len(CLASSES)).to(device)
    
    # Watchdog is standard CrossEntropy, Analyst is independent BCE (Multi-Label)
    criterion_binary = nn.CrossEntropyLoss(weight=binary_weights)
    criterion_fine = nn.BCEWithLogitsLoss(pos_weight=fine_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    print("\n" + "="*60)
    print(f"🧠 STARTING STRICT MULTI-LABEL TRAINING LOOP (>{CONFIDENCE_THRESHOLD*100}%)")
    print("="*60)

    best_binary_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        model.train()
        total_loss, bin_correct, fine_correct, total = 0.0, 0, 0, 0
        
        train_loop = tqdm(train_loader, desc="Training", leave=False)
        for skeletons, spatial, labels in train_loop:
            skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)
            
            # Watchdog Labels (0 or 1)
            binary_labels = (labels > 0).long().to(device)
            
            # Convert Analyst Labels to Independent Multi-Label Format
            fine_labels_onehot = F.one_hot(labels, num_classes=len(CLASSES)).float().to(device)

            optimizer.zero_grad()
            out_binary, out_fine = model(skeletons, spatial)
            
            loss_bin = criterion_binary(out_binary, binary_labels)
            loss_fine = criterion_fine(out_fine, fine_labels_onehot)
            
            # Maintain the 2.0x priority on finding the threat in the first place
            loss = (2.0 * loss_bin) + (1.0 * loss_fine)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)
            
            # Watchdog Accuracy (Just needs to pick 0 or 1)
            _, bin_pred = torch.max(out_binary.data, 1)
            bin_correct += (bin_pred == binary_labels).sum().item()
            
            # 🌟 STRICT Multi-Label Accuracy: Did the ground truth trigger > 80% confidence?
            fine_probs = torch.sigmoid(out_fine)
            fine_correct += (fine_probs[torch.arange(labels.size(0)), labels] > CONFIDENCE_THRESHOLD).sum().item()
            
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
                
                # 🌟 STRICT EVALUATION: Testing against the 80% threshold
                fine_probs = torch.sigmoid(out_fine)
                val_fine_correct += (fine_probs[torch.arange(labels.size(0)), labels] > CONFIDENCE_THRESHOLD).sum().item()
                
                val_total += labels.size(0)
                
        val_bin_acc = 100 * val_bin_correct / val_total
        val_fine_acc = 100 * val_fine_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f}")
        print(f"   🛡️ THREAT DETECTION (Binary) -> Train: {train_bin_acc:.2f}% | Val: {val_bin_acc:.2f}%")
        print(f"   🎯 STRICT THREAT CATCH (>80%)-> Train: {train_fine_acc:.2f}% | Val: {val_fine_acc:.2f}%")

        scheduler.step(val_bin_acc)

        if val_bin_acc > best_binary_acc:
            best_binary_acc = val_bin_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   🌟 New High-Security Model Saved! (Threat Detection: {best_binary_acc:.2f}%)")

if __name__ == "__main__":
    train_multilabel_model()