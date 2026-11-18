import os
import cv2
from tqdm import tqdm

# ================= CONFIGURATION =================
BASE_DIR = r"D:\Dataset_raw\Fall"

# Input Image Directories
TRAIN_IMG = os.path.join(BASE_DIR, "train", "images")
VALID_IMG = os.path.join(BASE_DIR, "valid", "images")
TEST_IMG = os.path.join(BASE_DIR, "test", "images")

# Input Label Directories (The AI reads these to filter out ADL)
TRAIN_LBL = os.path.join(BASE_DIR, "train", "labels")
VALID_LBL = os.path.join(BASE_DIR, "valid", "labels")
TEST_LBL = os.path.join(BASE_DIR, "test", "labels")

# Output Workspace Directories
OUT_TRAIN = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\Fall"
OUT_TEST = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test\Fall"

TARGET_FRAMES = 150
FPS = 30.0  
VIDEO_SIZE = (640, 640) # Verified mathematically from your README.roboflow.txt
# =================================================

def stitch_fall_dataset(img_dir, lbl_dir, out_dir, split_name):
    print(f"\n⚙️ Processing '{split_name}' Split...")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(img_dir):
        print(f"   ❌ Directory not found: {img_dir}")
        return
        
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    all_images.sort() # Ensure chronological sequence
    
    total_images = len(all_images)
    print(f"   📁 Found {total_images} frames.")
    
    if total_images == 0:
        return

    videos_created = 0
    discarded_adl = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for i in tqdm(range(0, total_images, TARGET_FRAMES), desc=f"   Stitching {split_name}"):
        chunk_imgs = all_images[i : i + TARGET_FRAMES]
        
        if len(chunk_imgs) < TARGET_FRAMES:
            break
            
        # --- SMART LABEL FILTERING ---
        contains_fall = False
        for img_file in chunk_imgs:
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_file)
            
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # '1' is the Fall class according to data.yaml
                        if line.startswith("1 "): 
                            contains_fall = True
                            break
            if contains_fall:
                break # We found a fall! No need to check the rest of the 150 frames.
                
        # If no fall happened in this 5-second window, throw it in the trash
        if not contains_fall:
            discarded_adl += 1
            continue 
            
        # --- RENDER THE PURE VIDEO ---
        video_filename = f"fall_{split_name}_{videos_created:03d}.mp4"
        out_path = os.path.join(out_dir, video_filename)
        out_video = cv2.VideoWriter(out_path, fourcc, FPS, VIDEO_SIZE)
        
        for img_file in chunk_imgs:
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            
            if (img.shape[1], img.shape[0]) != VIDEO_SIZE:
                img = cv2.resize(img, VIDEO_SIZE)
                
            out_video.write(img)
            
        out_video.release()
        videos_created += 1

    print(f"   ✅ Created {videos_created} pure 'Fall' videos.")
    print(f"   🗑️ Discarded {discarded_adl} 'ADL' (Normal) sequences to protect AI math.")

def main():
    print("🚀 INITIATING SMART LABEL-AWARE STITCHER...")
    
    # We map both Roboflow Train and Valid into our AI's Train folder
    stitch_fall_dataset(TRAIN_IMG, TRAIN_LBL, OUT_TRAIN, "train")
    stitch_fall_dataset(VALID_IMG, VALID_LBL, OUT_TRAIN, "valid_as_train")
    
    # We map Roboflow Test to our AI's Test folder
    stitch_fall_dataset(TEST_IMG, TEST_LBL, OUT_TEST, "test")
    
    print("\n" + "="*60)
    print("🎉 FALL DATASET SECURED AND FILTERED!")
    print("="*60)

if __name__ == "__main__":
    main()
    