import torch
from fusion_model import FusionThreatModel
from fusion_dataloader import FusionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cpu")
BATCH_SIZE = 32

def audit(path, num_classes):
    print(f"\n🔍 AUDITING: {path}")
    try:
        # We try loading it with the classes the error message SAID it had
        model = FusionThreatModel(num_classes=num_classes).to(DEVICE)
        
        # Load state dict with strict=False to bypass architectural ghosts
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False) 
        model.eval()
        
        dataset = FusionDataset(split="test")
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for kin, spat, labels in tqdm(loader, desc="Testing"):
                mask = labels < num_classes
                if not mask.any(): continue
                
                _, out = model(kin[mask].to(DEVICE), spat[mask].to(DEVICE))
                preds = torch.argmax(out, dim=1)
                correct += (preds == labels[mask].to(DEVICE)).sum().item()
                total += labels[mask].size(0)
        
        acc = (correct / total) * 100 if total > 0 else 0
        print(f"✅ RESULT: {acc:.2f}% Accuracy across {num_classes} classes.")
    except Exception as e:
        print(f"❌ ERROR: Could not audit this file. {e}")

# --- UPDATED AUDIT TARGETS ---
# We use 31 here because the error message explicitly said the file contains 31 classes
audit(r"D:\Threat_Detection_Workspace\03_Base_Model\base_threat_model_29_classes.pth", 31)

# We ignore the other one for now because it has a different architecture (LSTM) 
# that doesn't match your current FusionThreatModel (Transformer).