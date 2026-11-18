import os
import shutil
from tqdm import tqdm

# ================= CONFIGURATION =================
# Where your complex hierarchy currently lives
SOURCE_DIR = r"D:\Dataset_raw\NORMAL" 

# Where you want the clean, flat videos to go (Staging Area)
DEST_DIR = r"D:\Threat_Detection_Workspace\01_Raw_Data\Normal"
# =================================================

def flatten_dataset():
    print(f"🚀 INITIALIZING DATASET FLATTENING ALGORITHM...")
    
    # Ensure the destination directory exists
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Supported video formats
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Step 1: Count total videos for the progress bar
    total_videos = 0
    video_paths = []
    
    print("🔍 Scanning hierarchy for video files...")
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                video_paths.append(os.path.join(root, file))
                total_videos += 1
                
    print(f"✅ Found {total_videos} valid video files spread across sub-folders.")
    
    if total_videos == 0:
        print("⚠️ No videos found. Check your SOURCE_DIR path.")
        return

    # Step 2: Copy and Rename
    print(f"📂 Copying and renaming files to {DEST_DIR}...")
    
    counter = 1
    for old_path in tqdm(video_paths, desc="Processing Normal Data"):
        # Extract the file extension (e.g., .mp4)
        ext = os.path.splitext(old_path)[1].lower()
        
        # Create a clean, uniform file name (e.g., Normal_Video_0001.mp4)
        new_filename = f"Normal_Video_{counter:04d}{ext}"
        new_path = os.path.join(DEST_DIR, new_filename)
        
        # Copy the file (we use copy instead of move to protect your original raw data)
        shutil.copy2(old_path, new_path)
        counter += 1

    print("\n" + "="*60)
    print(f"🎉 SUCCESS! All {total_videos} videos are flattened and staged in: {DEST_DIR}")
    print("="*60)

if __name__ == "__main__":
    flatten_dataset()