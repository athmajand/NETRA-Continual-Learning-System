import os
import cv2
import numpy as np
import scipy.io

# ==============================
# THE STRICT 150-FRAME CONFIG
# ==============================
# Pointing to the RAW testing folders in your Downloads directory
VIDEOS_DIR = r"C:\Users\acer\Downloads\Suspicious Behavior\testing_videos"
MAT_DIR = r"C:\Users\acer\Downloads\Suspicious Behavior\testing_vol"

# Routing directly to the AI's workspace
OUTPUT_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train\SuspiciousBehavior"

TARGET_FRAMES = 150 # Strict rule for PyTorch
OUT_FPS = 30.0      # Forces the video to play smoothly

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_strict_150():
    print("🚀 INITIATING STRICT 150-FRAME EXTRACTION...")
    
    valid_videos = 0
    
    for file in os.listdir(VIDEOS_DIR):
        if not file.endswith(('.avi', '.mp4')):
            continue
            
        name = os.path.splitext(file)[0]
        video_file = os.path.join(VIDEOS_DIR, file)
        
        # Match the mathematical .mat file
        mat_file = os.path.join(MAT_DIR, f"vol{name}.mat")
        if not os.path.exists(mat_file):
            mat_file = os.path.join(MAT_DIR, f"{name}_vol.mat") # Fallback
            if not os.path.exists(mat_file):
                print(f"⚠️ Missing MAT for {name}, skipping.")
                continue

        # 1. Load the ground truth matrix
        data = scipy.io.loadmat(mat_file)
        vol = data["vol"]
        importance = vol.mean(axis=(0,1))
        
        # 2. Failsafe: Does the video even have 150 frames?
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < TARGET_FRAMES:
            print(f"⚠️ Video {name} is too short. Skipping.")
            cap.release()
            continue
            
        # 3. Slide the EXACT 150-frame window over the math
        scores = np.convolve(importance, np.ones(TARGET_FRAMES), mode="valid")
        start = np.argmax(scores)
        
        # 4. Extract the perfect sequence
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_file = os.path.join(OUTPUT_DIR, f"suspicious_avenue_{name}.mp4")
        writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (w,h))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        frames_written = 0
        for _ in range(TARGET_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1
            
        cap.release()
        writer.release()
        
        # 5. Quality Control Check
        if frames_written == TARGET_FRAMES:
            print(f"✅ Saved {name} -> 150 Frames Exactly")
            valid_videos += 1
        else:
            os.remove(out_file)
            print(f"❌ Failed to read 150 frames for {name}. Deleted corrupted file.")
            
    print(f"\n🎉 SUCCESSFULLY GENERATED {valid_videos} PERFECT 150-FRAME VIDEOS!")

if __name__ == "__main__":
    extract_strict_150()