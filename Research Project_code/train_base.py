import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fusion_dataloader import FusionDataset, CLASSES
from fusion_model import FusionThreatModel

# ================= CONFIGURATION =================
SAVE_DIR = r"D:\Threat_Detection_Workspace\03_Base_Model"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "base_threat_model_29_classes.pth")

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
# =================================================

def train_base_model():
    print("🚀 INITIALIZING 29-CLASS BASE MODEL TRAINING...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Hardware: {device.type.upper()}")

    print("⏳ Loading Datasets...")
    train_dataset = FusionDataset(split="train")
    test_dataset = FusionDataset(split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionThreatModel(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n" + "="*60)
    print("🧠 STARTING TRAINING LOOP")
    print("="*60)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for skeletons, spatial, labels in train_loader:
            skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(skeletons, spatial)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Quick validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for skeletons, spatial, labels in test_loader:
                skeletons, spatial, labels = skeletons.to(device), spatial.to(device), labels.to(device)
                outputs = model(skeletons, spatial)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   🌟 New Best Model Saved! (Accuracy: {best_acc:.2f}%)")

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print(f"💾 Final Best Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_base_model()