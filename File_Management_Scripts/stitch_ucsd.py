import os
import cv2
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
# The root folder of the UCSD dataset
INPUT_DIR = r"D:\Dataset_raw\UCSD Anomaly Detection Dataset\UCSD_Anomaly_Dataset.v1p2"
# Where the forged MP4s will be saved
OUTPUT_DIR = r"D:\Dataset_raw\UCSD_Videos"
FPS = 10.0
# ==============================

def forge_ucsd_videos():
    print("🔨 INITIATING UCSD VIDEO FORGE: Stitching .TIF frames into MP4s...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_videos_created = 0

    # Walk through the entire dataset
    for root, dirs, files in os.walk(INPUT_DIR):
        # We only care about folders that contain .TIF images
        tif_files = sorted([f for f in files if f.lower().endswith('.tif')])
        
        if not tif_files:
            continue
            
        # Get the dataset category (UCSDped1 or UCSDped2) and the specific folder name (e.g., Train001)
        path_parts = root.split(os.sep)
        
        # A safety check to ensure we are grabbing the right folder names
        if len(path_parts) >= 2:
            dataset_type = path_parts[-3] # e.g., UCSDped1
            folder_type = path_parts[-2]  # e.g., Train or Test
            sequence_name = path_parts[-1] # e.g., Test001
            
            # Name the output file clearly: e.g., "UCSDped1_Test_Test001.mp4"
            output_filename = f"{dataset_type}_{folder_type}_{sequence_name}.mp4"
        else:
            output_filename = f"{os.path.basename(root)}.mp4"

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Read the first frame to get the exact width and height
        first_frame_path = os.path.join(root, tif_files[0])
        first_frame = cv2.imread(first_frame_path)
        
        if first_frame is None:
            print(f"⚠️ Warning: Could not read {first_frame_path}. Skipping sequence.")
            continue
            
        height, width, layers = first_frame.shape
        
        # Initialize the OpenCV Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        
        print(f"\n🎬 Stitching sequence [{output_filename}] ({len(tif_files)} frames)...")
        
        for tif in tqdm(tif_files, desc="   Writing Frames", leave=False):
            frame_path = os.path.join(root, tif)
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
                
        out.release()
        total_videos_created += 1

    print("\n" + "="*60)
    print(f"✅ FORGE COMPLETE! Successfully created {total_videos_created} MP4 videos.")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    forge_ucsd_videos()