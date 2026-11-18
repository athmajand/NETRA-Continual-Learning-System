import os
import cv2
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIGURATION =================
TRAIN_IMAGES_DIR = r"D:\Dataset_raw\Fence_Climbing\Train"
TEST_IMAGES_DIR = r"D:\Dataset_raw\Fence_Climbing\Test"

TRAIN_OUT_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\FenceClimbing"
TEST_OUT_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test\FenceClimbing"

TARGET_FRAMES = 150
FPS = 30.0  
VIDEO_SIZE = (1280, 720) # Extracted directly from your diagnostic scan!
# =================================================

def stitch_fence_split(input_dir, output_dir, split_name):
    print(f"\n⚙️ Processing '{split_name}' Split...")
    os.makedirs(output_dir, exist_ok=True)
    
    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not all_images:
        print(f"   ❌ No images found in {input_dir}")
        return

    # 1. Group images by their original source video to prevent jump-cuts
    video_groups = defaultdict(list)
    for img_name in all_images:
        if "_frame_" in img_name:
            # Splits "Fence Climbing 3_frame_001778.jpg" into "Fence Climbing 3"
            prefix = img_name.split("_frame_")[0] 
        else:
            prefix = "unknown_source"
            
        video_groups[prefix].append(img_name)
        
    print(f"   📂 Detected {len(video_groups)} unique source camera angles.")

    # 2. Process each source video separately
    videos_created = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for prefix, frames in video_groups.items():
        frames.sort() # Ensure chronological order within this specific camera angle
        total_frames = len(frames)
        
        # Chop this specific sequence into 150-frame chunks
        for i in range(0, total_frames, TARGET_FRAMES):
            chunk = frames[i : i + TARGET_FRAMES]
            
            # Drop the remainder if it's less than 150
            if len(chunk) < TARGET_FRAMES:
                break 
                
            video_filename = f"fence_{split_name}_{videos_created:03d}.mp4"
            out_path = os.path.join(output_dir, video_filename)
            
            out_video = cv2.VideoWriter(out_path, fourcc, FPS, VIDEO_SIZE)
            
            for frame_file in chunk:
                img_path = os.path.join(input_dir, frame_file)
                img = cv2.imread(img_path)
                
                # Failsafe resize
                if (img.shape[1], img.shape[0]) != VIDEO_SIZE:
                    img = cv2.resize(img, VIDEO_SIZE)
                    
                out_video.write(img)
                
            out_video.release()
            videos_created += 1

    print(f"   ✅ Successfully created {videos_created} pure continuous videos for {split_name}.")

def main():
    print("🚀 INITIATING ADVANCED FENCE-CLIMBING STITCHER...")
    stitch_fence_split(TRAIN_IMAGES_DIR, TRAIN_OUT_DIR, "train")
    stitch_fence_split(TEST_IMAGES_DIR, TEST_OUT_DIR, "test")
    print("\n" + "="*60)
    print("🎉 FENCE CLIMBING SECURED FOR THE AI WORKSPACE!")
    print("="*60)

if __name__ == "__main__":
    main()