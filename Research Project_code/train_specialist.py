import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from fusion_model import FusionThreatModel
from fusion_dataloader import FusionDataset

# --- TARGETING THE RENAMED FILE ---
BASE_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\base_threat_model_31_classes.pth"
SAVE_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\specialist_32.pth"

device = torch.device("cpu")

print("🧠 Initializing 32-class architecture...")
model = FusionThreatModel(num_classes=32).to(device)

print(f"📖 Loading weights from {BASE_PATH}...")
try:
    temp_sd = torch.load(BASE_PATH, map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in temp_sd:
                if "analyst_head" in name:
                    param.data[:31] = temp_sd[name].data
                else:
                    param.copy_(temp_sd[name])
    print("✅ Weights grafted successfully.")
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit()

# FREEZE CORE: Only train the 32nd neuron
for param in model.parameters():
    param.requires_grad = False
model.analyst_head.weight.requires_grad = True
model.analyst_head.bias.requires_grad = True

optimizer = optim.Adam(model.analyst_head.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("📂 Syncing Training Data...")
dataset = FusionDataset(split="train")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model.eval() 
print("🚀 TRAINING SPECIALIST NEURON FOR CLASS 32...")

for epoch in range(1):
    pbar = tqdm(loader)
    for kin, spat, labels in pbar:
        mask = labels == 31  # Only train on the new "Package Theft" class
        if not mask.any(): continue
        
        optimizer.zero_grad()
        _, out = model(kin[mask].to(device), spat[mask].to(device))
        loss = criterion(out, labels[mask].to(device))
        loss.backward()
        
        # IRON SHIELD: Keep the 31 old neurons untouched
        if model.analyst_head.weight.grad is not None:
            model.analyst_head.weight.grad[:31] = 0
            model.analyst_head.bias.grad[:31] = 0
        
        optimizer.step()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

torch.save(model.state_dict(), SAVE_PATH)
print(f"\n✅ TRAINING COMPLETE. Specialist brain saved to: {SAVE_PATH}")