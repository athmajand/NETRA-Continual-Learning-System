import os
import shutil
import cv2
import random
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_DIR = r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\Vandalism"
TRAIN_DEST = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\Vandalism"
TEST_DEST = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test\Vandalism"

TRAIN_TARGET = 160
TEST_TARGET = 40
# =================================================

def is_video_playable(filepath):
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception:
        return False

def curate_data():
    print("🚀 INITIALIZING DEEP SCAN DATA CURATION...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Cannot find source directory: {SOURCE_DIR}")
        return

    os.makedirs(TRAIN_DEST, exist_ok=True)
    os.makedirs(TEST_DEST, exist_ok=True)

    # 🌟 THE FIX: Recursively search all nested sub-folders
    all_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # Check for videos regardless of uppercase/lowercase extensions
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                all_files.append(os.path.join(root, file))
                
    print(f"📁 Deep scan found {len(all_files)} total videos in the Vandalism directory tree.")
    
    if len(all_files) == 0:
        print("⚠️ No videos found! Please verify the folder path in SOURCE_DIR.")
        return

    random.shuffle(all_files)

    train_copied = 0
    test_copied = 0

    print("\n⚙️ Testing and migrating videos...")
    for src_path in tqdm(all_files, desc="Processing Vandalism"):
        if train_copied >= TRAIN_TARGET and test_copied >= TEST_TARGET:
            break
            
        if is_video_playable(src_path):
            # Extract the original filename and add a prefix to prevent overwriting 
            # in case two sub-folders have videos with the same name (like 'video1.mp4')
            original_name = os.path.basename(src_path)
            
            if train_copied < TRAIN_TARGET:
                safe_name = f"vand_train_{train_copied:03d}_{original_name}"
                dst_path = os.path.join(TRAIN_DEST, safe_name)
                shutil.copy2(src_path, dst_path)
                train_copied += 1
            elif test_copied < TEST_TARGET:
                safe_name = f"vand_test_{test_copied:03d}_{original_name}"
                dst_path = os.path.join(TEST_DEST, safe_name)
                shutil.copy2(src_path, dst_path)
                test_copied += 1

    print("\n" + "="*60)
    print("🎉 DATA CURATION COMPLETE!")
    print(f"✅ Healthy videos copied to Train: {train_copied}/{TRAIN_TARGET}")
    print(f"✅ Healthy videos copied to Test:  {test_copied}/{TEST_TARGET}")
    print("="*60)

if __name__ == "__main__":
    curate_data()