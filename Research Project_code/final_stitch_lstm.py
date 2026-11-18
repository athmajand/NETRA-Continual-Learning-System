import torch

# PATHS
BASE_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\base_threat_model_31_classes.pth"
SPEC_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\specialist_32.pth"
FINAL_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\NETRA_MK1_FINAL_32.pth"

print("🧵 INITIATING NEURAL SURGERY (TRANSFORMER NAMES)...")

base_sd = torch.load(BASE_PATH)
spec_sd = torch.load(SPEC_PATH)

try:
    # Grafting using the 'analyst_head' keys that were verified in your previous step
    base_sd['analyst_head.weight'] = torch.cat([
        base_sd['analyst_head.weight'], 
        spec_sd['analyst_head.weight'][31:32]
    ], dim=0)

    base_sd['analyst_head.bias'] = torch.cat([
        base_sd['analyst_head.bias'], 
        spec_sd['analyst_head.bias'][31:32]
    ], dim=0)

    torch.save(base_sd, FINAL_PATH)
    print(f"✅ STITCH SUCCESSFUL! Saved to: {FINAL_PATH}")

except KeyError as e:
    print(f"❌ ERROR: Key {e} not found. Printing available keys to help you:")
    print(base_sd.keys())