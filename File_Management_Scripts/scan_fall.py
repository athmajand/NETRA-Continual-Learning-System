import os
import cv2

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = r"D:\Dataset_raw\Fall"
VIDEO_DIR = os.path.join(BASE_DIR, "Raw_Video")
CSV_DIR = os.path.join(BASE_DIR, "Keypoints_CSV")
# ==============================

def scan_fall_dataset():
    print("🔍 INITIATING FALL DATASET RECONNAISSANCE...")
    
    # 1. Scan the Videos
    if os.path.exists(VIDEO_DIR):
        videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi'))]
        print(f"\n🎬 Found {len(videos)} video files in [Raw_Video].")
        
        if videos:
            print("   Checking frame counts for the first 5 videos:")
            for vid in videos[:5]:
                vid_path = os.path.join(VIDEO_DIR, vid)
                cap = cv2.VideoCapture(vid_path)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                print(f"     -> {vid}: {frames} frames")
    else:
        print(f"\n❌ ERROR: Could not find folder: {VIDEO_DIR}")

    # 2. Scan the CSVs
    if os.path.exists(CSV_DIR):
        csvs = [f for f in os.listdir(CSV_DIR) if f.lower().endswith('.csv')]
        print(f"\n📊 Found {len(csvs)} pre-extracted CSV files in [Keypoints_CSV].")
    else:
        print(f"\n❌ ERROR: Could not find folder: {CSV_DIR}")

    print("\n============================================================")
    print("🏁 RECONNAISSANCE COMPLETE")
    print("============================================================")

if __name__ == "__main__":
    scan_fall_dataset()
    