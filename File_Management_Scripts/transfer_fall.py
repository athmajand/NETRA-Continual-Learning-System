import os
import shutil
import random
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
SOURCE_DIR = r"D:\Dataset_raw\Fall\Raw_Video"
TRAIN_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\Fall"
TEST_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test\Fall"

SPLIT_RATIO = 0.8  # 80% Train, 20% Test
# ==============================

def secure_fall_dataset():
    print("🌊 ROUTING 3,140 FALL VIDEOS TO SUPER BRAIN WORKSPACE...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ ERROR: Could not find raw video folder at {SOURCE_DIR}")
        return

    # Grab all videos
    videos = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.mp4')]
    
    if not videos:
        print("❌ No .mp4 files found to move.")
        return
        
    print(f"✅ Found {len(videos)} videos. Shuffling and splitting...")
    
    # Shuffle for a perfectly balanced AI split
    random.shuffle(videos)
    split_index = int(len(videos) * SPLIT_RATIO)
    
    train_vids = videos[:split_index]
    test_vids = videos[split_index:]
    
    # Create the workspace directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    total_moved = 0
    
    # Execute the copy to Train
    for vid in tqdm(train_vids, desc="   -> Copying to TRAIN"):
        src = os.path.join(SOURCE_DIR, vid)
        dst = os.path.join(TRAIN_DIR, vid)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        total_moved += 1
        
    # Execute the copy to Test
    for vid in tqdm(test_vids, desc="   -> Copying to TEST"):
        src = os.path.join(SOURCE_DIR, vid)
        dst = os.path.join(TEST_DIR, vid)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        total_moved += 1

    print("\n" + "="*60)
    print(f"🎉 SUCCESS! Added {total_moved} Fall videos to your AI's memory banks.")
    print("="*60)

if __name__ == "__main__":
    secure_fall_dataset()