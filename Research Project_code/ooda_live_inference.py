import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from collections import deque
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog

from fusion_model import FusionThreatModel

# ================= CONFIGURATION =================
MODEL_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\ewc_continual_model_31_classes.pth"
FAST_TRACK_DB_PATH = r"D:\Threat_Detection_Workspace\fast_track_memory_db.pt"

MAX_FRAMES = 150 # 5-second rolling window
SIMILARITY_THRESHOLD = 0.85 # 85% mathematical match to trigger Fast-Track

CLASSES = [
    'Normal', 'Fighting', 'Burglary', 'Arson', 'Shooting',
    'Abuse', 'Arrest', 'Assault', 'Explosion', 'Fall', 
    'FenceClimbing', 'Kidnapping', 'Peeping', 'RoadAccidents', 'Robbery', 
    'Shoplifting', 'slap', 'Snatching', 'stab', 'Stealing', 
    'SuspiciousBehavior', 'Vandalism', 'Vehicle_Intrusion', 'medical_emergency', 
    'Weaponized', 'weapon_brandishing', 'intoxicated_behaviour', 'Crowd_Anomaly', 
    'atm_theft', 'Animal_Attack', 'Stone_Pelting' 
]
# =================================================

def select_video_source():
    """Pops up a command menu and file dialog to select the video source."""
    print("\n" + "="*50)
    print("🎯 OODA-SOC TACTICAL INTERFACE: SOURCE SELECTION")
    print("="*50)
    print("[1] Live Webcam (Real-time)")
    print("[2] Upload/Select Video File (.mp4)")
    print("="*50)
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '2':
        print("\n📂 Opening file explorer... Please select your video.")
        root = tk.Tk()
        root.withdraw() # Hide the main empty tkinter window
        root.attributes('-topmost', True) # Bring dialog to the front
        
        file_path = filedialog.askopenfilename(
            title="Select a Video for Inference",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        root.destroy()
        
        if file_path:
            print(f"✅ Selected Target: {file_path}")
            return file_path
        else:
            print("⚠️ No file selected. Defaulting to Live Webcam.")
            return 0
    else:
        print("✅ Selected Target: Live Webcam")
        return 0

def load_tactical_systems():
    device = torch.device("cpu") 
    print("\n⏳ [1/4] Loading ResNet50 Spatial Engine...")
    spatial_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    spatial_model = torch.nn.Sequential(*(list(spatial_model.children())[:-1]))
    spatial_model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("🧠 [2/4] Loading 31-Class EWC Deep Neural Network...")
    fusion_model = FusionThreatModel(num_classes=31)
    fusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    fusion_model.eval()

    print("⚡ [3/4] Loading Aegis Fast-Track Intelligence Cache...")
    try:
        fast_track_db = torch.load(FAST_TRACK_DB_PATH, map_location=device)
        print(f"   -> Armed with {len(fast_track_db)} rapid-response signatures.")
    except Exception as e:
        print("   -> Fast-Track DB not found. Running on Deep Network only.")
        fast_track_db = {}

    print("🦴 [4/4] Calibrating MediaPipe Kinematics...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    return spatial_model, preprocess, fusion_model, fast_track_db, pose, mp_drawing

def run_ooda_loop():
    # Prompt the user for the video source FIRST
    video_source = select_video_source()
    
    spatial_model, preprocess, fusion_model, fast_track_db, pose, mp_drawing = load_tactical_systems()
    
    skel_buffer = deque(maxlen=MAX_FRAMES)
    spat_buffer = deque(maxlen=MAX_FRAMES)
    
    cap = cv2.VideoCapture(video_source)
    print("\n🎥 TACTICAL HUD LIVE! Press 'q' to quit.")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("End of video stream reached.")
                break

            # --- OBSERVE (Data Ingestion) ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten()
                # Draw futuristic skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
            else:
                landmarks = np.zeros(132)
                
            skel_buffer.append(landmarks)

            input_tensor = preprocess(frame_rgb).unsqueeze(0)
            features = spatial_model(input_tensor).squeeze().numpy()
            spat_buffer.append(features)

            # Pad buffers for the first 5 seconds
            skel_input = list(skel_buffer)
            while len(skel_input) < MAX_FRAMES: skel_input.append(np.zeros(132))
            spat_input = list(spat_buffer)
            while len(spat_input) < MAX_FRAMES: spat_input.append(np.zeros(2048))

            skel_tensor = torch.FloatTensor(np.array(skel_input)).unsqueeze(0)
            spat_tensor = torch.FloatTensor(np.array(spat_input)).unsqueeze(0)
            spat_mean = torch.mean(spat_tensor, dim=1)

            # --- ORIENT (Dual-Path Processing) ---
            # Path A: The Deep Neural Network
            out_binary, out_fine = fusion_model(skel_tensor, spat_mean)
            prob_binary = F.softmax(out_binary, dim=1).squeeze()
            prob_fine = F.softmax(out_fine, dim=1).squeeze()
            
            threat_confidence = prob_binary[1].item() * 100
            top_class_idx = torch.argmax(prob_fine).item()
            top_class_name = CLASSES[top_class_idx]
            top_class_conf = prob_fine[top_class_idx].item() * 100

            # Path B: The Aegis Fast-Track Vector Extraction
            lstm_out, _ = fusion_model.lstm(skel_tensor)
            kinematic_features = lstm_out[:, -1, :] 
            spatial_features = fusion_model.spatial_fc(spat_mean)
            live_vector = torch.cat((kinematic_features, spatial_features), dim=1).squeeze()

            fast_track_triggered = False
            fast_track_name = ""
            fast_track_score = 0.0

            if fast_track_db:
                for name, saved_vector in fast_track_db.items():
                    saved_tensor = torch.FloatTensor(saved_vector)
                    similarity = F.cosine_similarity(live_vector.unsqueeze(0), saved_tensor.unsqueeze(0)).item()
                    if similarity > SIMILARITY_THRESHOLD:
                        fast_track_triggered = True
                        fast_track_name = name
                        fast_track_score = similarity * 100
                        break

            # --- DECIDE & ACT (Tactical Logic Tree & HUD) ---
            
            # 1. NORMAL CONDITION
            if threat_confidence < 50.0:
                status_color = (0, 255, 0) # Green
                status_text = "CONDITION: GREEN (SECURE)"
                detail_text = f"Watchdog: {threat_confidence:.1f}% Threat"
            
            # 2. KNOWN THREAT CONDITION
            elif threat_confidence >= 50.0 and top_class_conf >= 50.0:
                status_color = (0, 0, 255) # Red
                status_text = "CONDITION: RED (KNOWN THREAT)"
                detail_text = f"Analyst: {top_class_name} ({top_class_conf:.1f}%)"
            
            # 3. ANOMALY CONDITION (Threat detected, but Analyst is confused)
            else:
                # 3A: Caught by Fast-Track 1-Shot Memory!
                if fast_track_triggered:
                    status_color = (255, 0, 255) # Neon Purple/Pink
                    status_text = "CONDITION: FAST-TRACK OVERRIDE"
                    detail_text = f"Match: {fast_track_name} ({fast_track_score:.1f}%)"
                # 3B: True Unknown Anomaly (Send to Quarantine)
                else:
                    status_color = (0, 165, 255) # Orange
                    status_text = "CONDITION: YELLOW (UNKNOWN ANOMALY)"
                    detail_text = f"Action: Routing to Layer 2 Quarantine"

            # Draw HUD UI
            cv2.rectangle(frame, (10, 10), (600, 100), (0, 0, 0), -1) # Dark background box
            cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            cv2.putText(frame, detail_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("OODA-SOC Tactical Interface", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ooda_loop()