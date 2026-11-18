import os
import shutil
import random
from tqdm import tqdm

# ==============================
# CONFIGURATION: EXACT MAPPING
# ==============================
# Mapping the exact class name to your verified raw folders
SOURCE_MAP = {
    "Abuse": r"D:\Dataset_raw\UCF Crime\Anomaly_Dataset\Anomaly_Videos\Anomaly-Videos-Part-1\Abuse",
    "Arrest": r"D:\Dataset_raw\UCF Crime\Anomaly_Dataset\Anomaly_Videos\Anomaly-Videos-Part-1\Arrest",
    "Assault": r"D:\Dataset_raw\UCF Crime\Anomaly_Dataset\Anomaly_Videos\Anomaly-Videos-Part-1\Assault",
    "Explosion": r"D:\Dataset_raw\UCF Crime\Anomaly_Dataset\Anomaly_Videos\Anomaly-Videos-Part-2\Explosion",
    "Snatching": r"D:\Dataset_raw\snatching",
    "Peeping": r"D:\Dataset_raw\Peeping"
}

TRAIN_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train"
TEST_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test"

SPLIT_RATIO = 0.8  # 80% Train, 20% Test
# ==============================

def open_the_floodgates():
    print("🌊 OPENING THE FLOODGATES: Transferring mapped videos (No Frame Limits)...")
    
    total_moved = 0
    
    for cls, source_path in SOURCE_MAP.items():
        print(f"\n🔍 Processing [{cls}] from: {source_path}")
        
        if not os.path.exists(source_path):
            print(f"   ❌ Source folder not found! Check path for {cls}...")
            continue
            
        # Find all MP4s directly in this specific folder
        video_paths = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.lower().endswith('.mp4')]
                        
        if not video_paths:
            print(f"   ❌ Could not find any .mp4 files in {source_path}")
            continue
            
        print(f"   ✅ Found {len(video_paths)} videos. Splitting and copying...")
        
        # Shuffle for a fair AI split
        random.shuffle(video_paths)
        split_index = int(len(video_paths) * SPLIT_RATIO)
        
        train_vids = video_paths[:split_index]
        test_vids = video_paths[split_index:]
        
        # Create destination folders
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)
        
        # Copy to Train
        for vid in tqdm(train_vids, desc=f"   -> {cls} to TRAIN"):
            shutil.copy2(vid, os.path.join(TRAIN_DIR, cls, os.path.basename(vid)))
            total_moved += 1
            
        # Copy to Test
        for vid in tqdm(test_vids, desc=f"   -> {cls} to TEST"):
            shutil.copy2(vid, os.path.join(TEST_DIR, cls, os.path.basename(vid)))
            total_moved += 1

    print("\n" + "="*60)
    print(f"🎉 TRANSFER COMPLETE! Successfully moved {total_moved} raw videos into the workspace.")
    print("="*60)

if __name__ == "__main__":
    open_the_floodgates()