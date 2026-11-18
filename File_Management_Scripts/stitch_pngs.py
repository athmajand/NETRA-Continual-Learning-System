import os
import cv2
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_DIR = r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes\Vandalism"
TRAIN_DEST = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\Vandalism"
TEST_DEST = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test\Vandalism"
# =================================================

def build_videos():
    print("🚀 INITIALIZING PNG-TO-VIDEO STITCHER...")
    
    os.makedirs(TRAIN_DEST, exist_ok=True)
    os.makedirs(TEST_DEST, exist_ok=True)

    # 1. Gather all PNGs
    all_pngs = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_pngs.append(os.path.join(root, f))
                
    # 🌟 CRITICAL: Sort them so the frames stay in exact chronological order
    all_pngs.sort()
    
    if len(all_pngs) == 0:
        print("❌ No images found!")
        return
        
    print(f"📁 Found {len(all_pngs)} frames. Stitching them back into MP4 videos...")
    
    # 2. Chunk them into 150-frame videos (perfectly matching our LSTM window!)
    FRAMES_PER_VIDEO = 150
    FPS = 30.0
    
    total_videos_created = 0
    
    # Read the first image to lock in the master video resolution
    first_frame = cv2.imread(all_pngs[0])
    height, width, layers = first_frame.shape
    video_size = (width, height)
    
    for i in tqdm(range(0, len(all_pngs), FRAMES_PER_VIDEO), desc="Stitching Videos"):
        chunk = all_pngs[i : i + FRAMES_PER_VIDEO]
        
        # If the chunk is smaller than 150 frames, skip it (we need uniform data)
        if len(chunk) < FRAMES_PER_VIDEO:
            break
            
        # Route to Train (160 videos) or Test (40 videos)
        if total_videos_created < 160:
            out_path = os.path.join(TRAIN_DEST, f"vand_stitched_{total_videos_created:03d}.mp4")
        elif total_videos_created < 200:
            out_path = os.path.join(TEST_DEST, f"vand_stitched_{total_videos_created:03d}.mp4")
        else:
            break # We hit our 200 video quota!
            
        # Initialize the OpenCV Video Engine
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, FPS, video_size)
        
        for frame_path in chunk:
            img = cv2.imread(frame_path)
            # Failsafe: if an image randomly changes size, force it to match the master video size
            if (img.shape[1], img.shape[0]) != video_size:
                img = cv2.resize(img, video_size)
            out_video.write(img)
            
        out_video.release()
        total_videos_created += 1

    print("\n" + "="*60)
    print("🎉 STITCHING COMPLETE!")
    print(f"✅ Reconstructed {total_videos_created} highly-accurate MP4 Videos.")
    print("="*60)

if __name__ == "__main__":
    build_videos()