import os
import cv2
from tqdm import tqdm
from glob import glob

# ==============================
# CONFIGURATION
# ==============================
TARGET_DIR = r"D:\Dataset_raw\SHANGHAI"
OUTPUT_DIR = r"D:\Dataset_raw\ShanghaiTech_Videos"
FPS = 30.0
# ==============================

def forge_videos():
    print("🔨 INITIATING VIDEO FORGE: Stitching 173,000 frames into MP4s...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Walk through the directory to find folders containing JPGs
    for root, dirs, files in os.walk(TARGET_DIR):
        # Skip the label folder
        if "label" in root.lower():
            continue
            
        # Find all JPGs in the current folder
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue # No images here, move to the next folder

        # Sort images numerically (001.jpg, 002.jpg, etc.)
        images.sort()
        
        folder_name = os.path.basename(root)
        output_path = os.path.join(OUTPUT_DIR, f"{folder_name}.mp4")
        
        if os.path.exists(output_path):
            continue # Skip if we already stitched this video
            
        print(f"\n🎬 Stitching sequence [{folder_name}] ({len(images)} frames)...")
        
        # Read the first image to get the dimensions (Width, Height)
        first_img_path = os.path.join(root, images[0])
        frame = cv2.imread(first_img_path)
        if frame is None:
            print(f"   ❌ Error reading {first_img_path}")
            continue
            
        height, width, layers = frame.shape
        size = (width, height)
        
        # Initialize OpenCV Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(output_path, fourcc, FPS, size)
        
        # Write each frame to the video
        for img_name in tqdm(images, desc="   Writing Frames"):
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                out.write(img)
                
        out.release() # Save the video

    print("\n" + "="*60)
    print(f"✅ FORGE COMPLETE! MP4 videos saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    forge_videos()