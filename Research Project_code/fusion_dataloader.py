import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ================= CONFIGURATION =================
SKELETON_DIR = r"D:\Threat_Detection_Workspace\02_Feature_Extraction\Pose_Features_Granular"
SPATIAL_DIR = r"D:\Threat_Detection_Workspace\02_Feature_Extraction\Spatial_Features_Granular"

CLASSES = [
    'Normal', 'Fighting', 'Burglary', 'Arson', 'Shooting',
    'Abuse', 'Arrest', 'Assault', 'Explosion', 'Fall', 
    'FenceClimbing', 'Kidnapping', 'Peeping', 'RoadAccidents', 'Robbery', 
    'Shoplifting', 'slap', 'Snatching', 'stab', 'Stealing', 
    'SuspiciousBehavior', 'Vandalism', 'Vehicle_Intrusion', 'medical_emergency', 
    'Weaponized', 'weapon_brandishing', 'intoxicated_behaviour', 'Crowd_Anomaly', 
    'atm_theft', 'Animal_Attack', 'Stone_Pelting' , 'Package_Theft' # 🌟 The 2 New Classes Added
]


MAX_FRAMES = 150  # 5-second sliding window
# =================================================

class FusionDataset(Dataset):
    def __init__(self, split="train"):
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}
        self.samples = [] 
        
        skel_split_dir = os.path.join(SKELETON_DIR, split)
        spat_split_dir = os.path.join(SPATIAL_DIR, split)
        
        print(f"🔍 Synchronizing {split.upper()} Data...")
        
        for cls_name in CLASSES:
            skel_cls_dir = os.path.join(skel_split_dir, cls_name)
            spat_cls_dir = os.path.join(spat_split_dir, cls_name)
            
            if not os.path.exists(skel_cls_dir) or not os.path.exists(spat_cls_dir):
                continue
                
            for file in os.listdir(skel_cls_dir):
                if file.endswith('.npy'):
                    skel_path = os.path.join(skel_cls_dir, file)
                    spat_path = os.path.join(spat_cls_dir, file)
                    label = self.class_to_idx[cls_name]
                    
                    if not os.path.exists(spat_path): continue
                    
                    try:
                        total_frames = np.load(skel_path, mmap_mode='r').shape[0]
                        if total_frames == 0: continue
                            
                        # The Sliding Window
                        for start_frame in range(0, total_frames, MAX_FRAMES):
                            self.samples.append((skel_path, start_frame, spat_path, label))
                            
                    except Exception as e:
                        pass

        print(f"✅ {split.capitalize()} Dataset ready! {len(self.samples)} sliding-window slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skel_path, start_frame, spat_path, label = self.samples[idx]
        end_frame = start_frame + MAX_FRAMES
        
        # --- 1. SKELETON DATA ---
        skel_tensor = torch.FloatTensor(np.load(skel_path))
        if skel_tensor.dim() == 1: skel_tensor = skel_tensor.unsqueeze(0)
            
        skel_chunk = skel_tensor[start_frame:end_frame, :]
        if skel_chunk.shape[0] < MAX_FRAMES:
            padding = torch.zeros(MAX_FRAMES - skel_chunk.shape[0], 132)
            skel_chunk = torch.cat((skel_chunk, padding), dim=0)
            
        # --- 2. SPATIAL DATA ---
        spat_tensor = torch.FloatTensor(np.load(spat_path))
        if spat_tensor.dim() == 1: spat_tensor = spat_tensor.unsqueeze(0)
            
        spat_chunk = spat_tensor[start_frame:end_frame, :]
        if spat_chunk.numel() == 0:
            spat_mean = torch.zeros(2048)
        else:
            spat_mean = torch.mean(spat_chunk, dim=0)
            
        return skel_chunk, spat_mean, torch.tensor(label, dtype=torch.long)