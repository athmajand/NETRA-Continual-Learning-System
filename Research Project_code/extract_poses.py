import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ================= CONFIGURATION =================
BASE_DATASET_PATH = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular"
BASE_OUTPUT_PATH = r"D:\Threat_Detection_Workspace\02_Feature_Extraction\Pose_Features_Granular"

# 🌟 TARGETING THE NEW ZERO-DAY THREAT 🌟
CLASSES = ['Package_Theft']
# =================================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten()
        else:
            landmarks = np.zeros(33 * 4)
            
        frames_data.append(landmarks)

    cap.release()
    return np.array(frames_data)

def process_poses():
    print("🚀 STARTING CLEAN-SLATE POSE EXTRACTION (TRAIN & TEST)...")
    
    for split in ['train', 'test']:
        print(f"\n" + "="*50)
        print(f"📂 PROCESSING SPLIT: {split.upper()}")
        print("="*50)
        
        for cls in CLASSES:
            class_path = os.path.join(BASE_DATASET_PATH, split, cls)
            out_class_path = os.path.join(BASE_OUTPUT_PATH, split, cls)
            
            if not os.path.exists(class_path):
                print(f"⚠️ Missing raw data for {split}/{cls}")
                continue
                
            os.makedirs(out_class_path, exist_ok=True)
            videos = [v for v in os.listdir(class_path) if v.endswith(('.mp4', '.avi'))]
            
            if len(videos) > 0:
                print(f"\n🦴 Extracting Poses -> [{cls}] ({len(videos)} videos)")
            
            for video_name in tqdm(videos, desc=f"{cls}"):
                video_path = os.path.join(class_path, video_name)
                npy_filename = os.path.splitext(video_name)[0] + '.npy'
                npy_path = os.path.join(out_class_path, npy_filename)
                
                if os.path.exists(npy_path): continue
                
                try:
                    pose_data = extract_pose_from_video(video_path)
                    np.save(npy_path, pose_data)
                except Exception as e:
                    print(f"   ⚠️ Error processing {video_name}: {e}")

if __name__ == "__main__":
    process_poses()