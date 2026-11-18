import torch
import torch.nn as nn
from fusion_model import FusionThreatModel

print("🧬 INITIATING HARD GRAFT PROTOCOL...")

# ================= PATHS =================
BASE_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\base_threat_model_29_classes.pth"
TRAINED_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\evolved_threat_model_32_classes_FINAL.pth"
SAVE_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\evolved_threat_model_32_classes_PERFECT.pth"
# =========================================

device = torch.device("cpu")

print("🧠 Loading pure, uncorrupted base model...")
pure_model = FusionThreatModel(num_classes=31).to(device)
pure_model.load_state_dict(torch.load(BASE_PATH, map_location=device))

print("🔪 Extracting the 32nd neuron from the 12-hour training run...")
trained_model = FusionThreatModel(num_classes=32).to(device)
trained_model.load_state_dict(torch.load(TRAINED_PATH, map_location=device))

# 1. Prepare the new perfect layer
perfect_weight = torch.zeros((32, 128))
perfect_bias = torch.zeros(32)

# 2. Copy the perfect core memories (Classes 0 to 30)
perfect_weight[:31] = pure_model.analyst_head.weight.data
perfect_bias[:31] = pure_model.analyst_head.bias.data

# 3. Copy the highly-trained new threat (Class 31)
new_weight = trained_model.analyst_head.weight.data[31]
new_bias = trained_model.analyst_head.bias.data[31]

# 4. Apply Volume Correction so it doesn't scream over the old neurons
old_vol = perfect_weight[:31].abs().mean()
new_vol = new_weight.abs().mean()
scale = old_vol / (new_vol + 1e-8)

perfect_weight[31] = new_weight * scale
perfect_bias[31] = perfect_bias[:31].mean()

# 5. Perform the final surgical attachment
pure_model.analyst_head = nn.Linear(in_features=128, out_features=32)
with torch.no_grad():
    pure_model.analyst_head.weight.data = perfect_weight
    pure_model.analyst_head.bias.data = perfect_bias

# 6. Save the masterpiece
torch.save(pure_model.state_dict(), SAVE_PATH)
print(f"\n✅ PERFECT GRAFT SAVED TO: {SAVE_PATH}")