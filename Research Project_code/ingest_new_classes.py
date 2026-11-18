import os
import shutil
import random

# ================= CONFIGURATION =================
# Pointing to the parent folder of where Hugging Face put the files
SOURCE_DIR = r"D:\Dataset_raw\Package_Theft"

# The destination where your main dataset lives
TRAIN_DEST = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train"
TEST_DEST = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test"

# Mapping the Hugging Face "videos" folder to your clean class name
NEW_CLASSES = {
    "videos": "Package_Theft"  # This will create a "Package_Theft" folder in your train/test directories
}

SPLIT_RATIO = 0.80 # 80% Train, 20% Test
# =================================================

def ingest_data():
    print("🚀 INITIATING DATA INGESTION FOR CONTINUAL LEARNING...")
    
    for raw_name, clean_name in NEW_CLASSES.items():
        src_class_path = os.path.join(SOURCE_DIR, raw_name)
        
        if not os.path.exists(src_class_path):
            print(f"⚠️ Could not find folder: {src_class_path}")
            continue
            
        videos = [v for v in os.listdir(src_class_path) if v.lower().endswith('.mp4')]
        random.shuffle(videos) # Shuffle to ensure random distribution
        
        train_split_idx = int(len(videos) * SPLIT_RATIO)
        train_vids = videos[:train_split_idx]
        test_vids = videos[train_split_idx:]
        
        # Create destination folders
        os.makedirs(os.path.join(TRAIN_DEST, clean_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DEST, clean_name), exist_ok=True)
        
        print(f"\n📂 Processing [{clean_name}] ({len(videos)} total videos)")
        
        # Copy to Train
        for vid in train_vids:
            shutil.copy(os.path.join(src_class_path, vid), os.path.join(TRAIN_DEST, clean_name, vid))
        print(f"   --> Sent {len(train_vids)} videos to TRAIN")
        
        # Copy to Test
        for vid in test_vids:
            shutil.copy(os.path.join(src_class_path, vid), os.path.join(TEST_DEST, clean_name, vid))
        print(f"   --> Sent {len(test_vids)} videos to TEST")

    print("\n✅ INGESTION COMPLETE! New class is ready for feature extraction.")

if __name__ == "__main__":
    ingest_data()