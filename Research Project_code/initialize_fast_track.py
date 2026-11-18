import os
import torch
import numpy as np
from fusion_model import FusionThreatModel

# ================= CONFIGURATION =================
BASE_MODEL_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\ewc_continual_model_31_classes.pth"
FAST_TRACK_DB_PATH = r"D:\Threat_Detection_Workspace\fast_track_memory_db.pt"

# 🌟 THE FIX: Pointing exactly to the train split folders
POSE_DIR = r"D:\Threat_Detection_Workspace\02_Feature_Extraction\Pose_Features_Granular\train\Stone_Pelting"
SPATIAL_DIR = r"D:\Threat_Detection_Workspace\02_Feature_Extraction\Spatial_Features_Granular\train\Stone_Pelting"

# We are simulating the Admin tagging ONE specific video as a new threat
TARGET_CLASS_NAME = "Stone_Pelting_1_Shot"
# =================================================

def build_fast_track_memory():
    print("🛡️ INITIATING FAST-TRACK MEMORY COMPILER...")
    device = torch.device("cpu")
    
    # 1. Load the Brain
    print("🧠 Loading Neural Cortex...")
    model = FusionThreatModel(num_classes=31).to(device)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    model.eval()

    # 2. Grab the first available extracted video for Stone Pelting
    pose_files = os.listdir(POSE_DIR)
    if not pose_files:
        print("❌ No pose features found! Make sure extraction finished.")
        return
        
    target_filename = pose_files[0] # Just grab the first one
    print(f"🎯 Admin Selected Anomaly: {target_filename}")

    pose_path = os.path.join(POSE_DIR, target_filename)
    spatial_path = os.path.join(SPATIAL_DIR, target_filename)

    # 3. Load the raw numpy arrays
    skel_data = np.load(pose_path)
    spat_data = np.load(spatial_path)

    # Convert to tensors and add batch dimension
    skel_tensor = torch.FloatTensor(skel_data).unsqueeze(0).to(device)
    spat_tensor = torch.FloatTensor(spat_data).unsqueeze(0).to(device)
    
    # We condense the spatial data over time just like in training
    spat_mean = torch.mean(spat_tensor, dim=1)

    # 4. 🔪 EXTRACT THE CORE MATHEMATICAL VECTOR
    print("🧬 Extracting 256-Dimensional Feature Vector...")
    with torch.no_grad():
        # Pass through LSTM
        lstm_out, _ = model.lstm(skel_tensor)
        kinematic_features = lstm_out[:, -1, :] 
        
        # Pass through Spatial FC
        spatial_features = model.spatial_fc(spat_mean)
        
        # This is the "Mathematical Essence" of the video right before the decision heads
        fused_vector = torch.cat((kinematic_features, spatial_features), dim=1)

    # 5. Save to the Fast-Track Database
    # We use a simple dictionary where Key = Threat Name, Value = Vector
    fast_track_db = {
        TARGET_CLASS_NAME: fused_vector.squeeze().numpy()
    }
    
    torch.save(fast_track_db, FAST_TRACK_DB_PATH)
    print("\n" + "="*60)
    print(f"✅ FAST-TRACK MEMORY SAVED: {FAST_TRACK_DB_PATH}")
    print(f"   Stored Identity: [{TARGET_CLASS_NAME}]")
    print(f"   Vector Shape: {fused_vector.shape}")
    print("="*60)
    print("Layer 2 Intelligence Cache is now armed and ready for Live Inference.")

if __name__ == "__main__":
    build_fast_track_memory()