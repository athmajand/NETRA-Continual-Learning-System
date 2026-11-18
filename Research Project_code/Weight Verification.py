import torch
BASE_MODEL = torch.load(r"D:\Threat_Detection_Workspace\03_Base_Model\base_threat_model_31_classes.pth")
SPECIALIST = torch.load(r"D:\Threat_Detection_Workspace\03_Base_Model\specialist_32.pth")

# Pick a random layer from the 'Old' part of the brain
old_weight_base = BASE_MODEL['spatial_fc.0.weight']
old_weight_spec = SPECIALIST['spatial_fc.0.weight']

# If this is 0.0, the Iron Shield worked!
diff = torch.sum(old_weight_base - old_weight_spec).item()
if diff == 0.0:
    print("✅ SUCCESS: The 31 original classes are PERFECTLY preserved.")
else:
    print(f"❌ FAILURE: The specialist training corrupted the old brain by {diff} magnitude.")