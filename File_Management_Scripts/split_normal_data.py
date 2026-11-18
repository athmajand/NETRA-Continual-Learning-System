import os
import shutil
import random
from tqdm import tqdm

# ================= CONFIGURATION =================
# The source folder containing your videos
# Change this to D:\Threat_Detection_Workspace\01_Raw_Data\Normal if you want to use the flattened videos!
SOURCE_DIR = r"D:\Dataset_raw\NORMAL" 

# Destination folders (Notice the \Normal appended at the end to maintain class structure)
TRAIN_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\Normal"
TEST_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test\Normal"

# Train/Test Split Ratio (80%)
TRAIN_RATIO = 0.80
# =================================================

def perform_split():
    print(f"🚀 INITIALIZING 80/20 INGESTION PROTOCOL...")
    
    # Ensure destination directories exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = []
    
    print("🔍 Scanning source directory for videos...")
    # Walk through the directory (handles both flat and nested structures)
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                video_paths.append(os.path.join(root, file))
                
    total_videos = len(video_paths)
    print(f"✅ Found {total_videos} videos ready for ingestion.")
    
    if total_videos == 0:
        print("⚠️ No videos found. Halting execution.")
        return

    # 🎲 Algorithmic Shuffle to ensure random distribution
    print("🎲 Shuffling data to prevent chronological bias...")
    random.shuffle(video_paths)
    
    # Calculate exact split numbers
    train_count = int(total_videos * TRAIN_RATIO)
    test_count = total_videos - train_count
    
    train_videos = video_paths[:train_count]
    test_videos = video_paths[train_count:]
    
    print(f"📊 Split math complete: {train_count} for Training (80%) | {test_count} for Testing (20%)")

    # Move Training Data
    print("\n📂 Copying Training Data...")
    for video in tqdm(train_videos, desc="Train Split"):
        filename = os.path.basename(video)
        dest_path = os.path.join(TRAIN_DIR, filename)
        # Using copy2 to preserve metadata and prevent destroying your original dataset
        if not os.path.exists(dest_path): 
            shutil.copy2(video, dest_path)

    # Move Testing Data
    print("\n📂 Copying Testing Data...")
    for video in tqdm(test_videos, desc="Test Split"):
        filename = os.path.basename(video)
        dest_path = os.path.join(TEST_DIR, filename)
        if not os.path.exists(dest_path):
            shutil.copy2(video, dest_path)

    print("\n" + "="*60)
    print("🎉 INGESTION SUCCESSFUL!")
    print(f"Training Directory: {TRAIN_DIR} ({len(os.listdir(TRAIN_DIR))} files)")
    print(f"Testing Directory:  {TEST_DIR} ({len(os.listdir(TEST_DIR))} files)")
    print("="*60)

if __name__ == "__main__":
    # Optional: Set a seed if you want the "random" shuffle to be identical every time you run it
    random.seed(42) 
    perform_split()