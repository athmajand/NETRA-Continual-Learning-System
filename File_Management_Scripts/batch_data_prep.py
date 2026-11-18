import os
import cv2
import math
from tqdm import tqdm

# ================= CONFIGURATION =================
# The 6 new raw folders you want to process
RAW_FOLDERS = {
    "RoadAccidents": r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\RoadAccidents",
    "Robbery": r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\Robbery",
    "Shoplifting": r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\Shoplifting",
    "slap": r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\slap",
    "stab": r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\stab",
    "Stealing": r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\Stealing"
}

TRAIN_BASE = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train"
TEST_BASE = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test"

FRAMES_PER_VIDEO = 150
FPS = 30.0
# =================================================

def process_batch():
    print("🚀 INITIALIZING MASTER BATCH DATA PREPARATION...")

    for class_name, source_dir in RAW_FOLDERS.items():
        print(f"\n" + "="*50)
        print(f"🔍 SCANNING: {class_name}")
        
        if not os.path.exists(source_dir):
            print(f"   ❌ Folder not found! Skipping...")
            continue
            
        train_dest = os.path.join(TRAIN_BASE, class_name)
        test_dest = os.path.join(TEST_BASE, class_name)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        # 1. Gather and sort all images chronologically
        all_pngs = []
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_pngs.append(os.path.join(root, f))
                    
        all_pngs.sort()
        
        if len(all_pngs) == 0:
            print(f"   ⚠️ No images found in {class_name}. Skipping...")
            continue
            
        # Calculate how many videos we can make
        total_possible_videos = len(all_pngs) // FRAMES_PER_VIDEO
        if total_possible_videos == 0:
            print(f"   ⚠️ Not enough frames to make even one 150-frame video. Skipping...")
            continue
            
        print(f"   📁 Found {len(all_pngs)} frames. Stitching into ~{total_possible_videos} videos...")

        # 80/20 Train/Test Split
        train_quota = math.floor(total_possible_videos * 0.80)
        
        first_frame = cv2.imread(all_pngs[0])
        height, width, _ = first_frame.shape
        video_size = (width, height)
        
        videos_created = 0
        
        for i in tqdm(range(0, len(all_pngs), FRAMES_PER_VIDEO), desc=f"Stitching {class_name}"):
            chunk = all_pngs[i : i + FRAMES_PER_VIDEO]
            if len(chunk) < FRAMES_PER_VIDEO:
                break
                
            # Route to train or test based on quota
            if videos_created < train_quota:
                out_path = os.path.join(train_dest, f"{class_name.lower()}_{videos_created:03d}.mp4")
            else:
                out_path = os.path.join(test_dest, f"{class_name.lower()}_{videos_created:03d}.mp4")
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(out_path, fourcc, FPS, video_size)
            
            for frame_path in chunk:
                img = cv2.imread(frame_path)
                if (img.shape[1], img.shape[0]) != video_size:
                    img = cv2.resize(img, video_size)
                out_video.write(img)
                
            out_video.release()
            videos_created += 1

        print(f"   ✅ Finished {class_name}: {videos_created} videos created.")

    print("\n" + "="*60)
    print("🎉 MASTER BATCH PREPARATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    process_batch()