import os
import cv2
from tqdm import tqdm

# ================= CONFIGURATION =================
# 1. The exact paths to your newly downloaded Roboflow dataset
TRAIN_IMAGES_DIR = r"D:\Dataset_raw\kidnap__Annotation.v1i.yolov8\train\images"
VALID_IMAGES_DIR = r"D:\Dataset_raw\kidnap__Annotation.v1i.yolov8\valid\images"

# 2. Where your AI expects the training and testing videos to be
TRAIN_OUT_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\Kidnapping"
TEST_OUT_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test\Kidnapping"

TARGET_FRAMES = 150
FPS = 30.0  
VIDEO_SIZE = (640, 640) # Standard Roboflow resize dimension
# =================================================

def stitch_split(input_dir, output_dir, split_name):
    print(f"\n⚙️ Processing '{split_name}' Split...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"   ❌ Directory not found: {input_dir}")
        return
        
    # Grab images and sort alphabetically to ensure time sequence is correct
    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    all_images.sort() 
    
    total_images = len(all_images)
    print(f"   📁 Found {total_images} sequential frames.")
    
    if total_images == 0:
        return

    videos_created = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Loop through the images in exact 150-frame chunks
    for i in tqdm(range(0, total_images, TARGET_FRAMES), desc=f"   Stitching {split_name} Videos"):
        chunk = all_images[i : i + TARGET_FRAMES]
        
        # STRICT RULE: Discard the leftover frames at the end if they don't equal exactly 150
        if len(chunk) < TARGET_FRAMES:
            break
            
        video_filename = f"kidnap_{split_name}_{videos_created:03d}.mp4"
        out_path = os.path.join(output_dir, video_filename)
        
        out_video = cv2.VideoWriter(out_path, fourcc, FPS, VIDEO_SIZE)
        
        for frame_file in chunk:
            img_path = os.path.join(input_dir, frame_file)
            img = cv2.imread(img_path)
            
            # Failsafe resize just in case
            if (img.shape[1], img.shape[0]) != VIDEO_SIZE:
                img = cv2.resize(img, VIDEO_SIZE)
                
            out_video.write(img)
            
        out_video.release()
        videos_created += 1

    print(f"   ✅ Created {videos_created} videos for {split_name}.")

def main():
    print("🚀 INITIATING ROBOFLOW DATASET STITCHER...")
    # Convert the Roboflow 'train' images to our workspace 'train' videos
    stitch_split(TRAIN_IMAGES_DIR, TRAIN_OUT_DIR, "train")
    
    # Convert the Roboflow 'valid' images to our workspace 'test' videos
    stitch_split(VALID_IMAGES_DIR, TEST_OUT_DIR, "test")
    
    print("\n" + "="*60)
    print("🎉 ALL KIDNAPPING VIDEOS SUCCESSFULLY GENERATED!")
    print("="*60)

if __name__ == "__main__":
    main()