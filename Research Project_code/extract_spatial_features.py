import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# ================= CONFIGURATION =================
BASE_DATASET_PATH = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular"
BASE_OUTPUT_PATH = r"D:\Threat_Detection_Workspace\02_Feature_Extraction\Spatial_Features_Granular"

# CHANGED THIS TO TARGET YOUR NEW ZERO-DAY THREAT
CLASSES = ['Package_Theft']
# =================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    model.eval() 
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess

def extract_spatial_from_video(video_path, model, preprocess):
    cap = cv2.VideoCapture(video_path)
    frames_data = []

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame_rgb)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            features = model(input_batch)
            features = features.squeeze().cpu().numpy() 
            frames_data.append(features)

    cap.release()
    return np.array(frames_data)

def process_dataset():
    print(f"🚀 STARTING DEEP-LOGGING SPATIAL EXTRACTION on device: {device}...")
    model, preprocess = get_feature_extractor()
    
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
                print(f"\n⚙️ Extracting Environment -> [{cls}] ({len(videos)} videos)")
            
            # --- UPGRADED DEEP LOGGING ---
            skipped_count = 0
            extracted_count = 0
            pbar = tqdm(videos, desc=f"{cls}")
            
            for video_name in pbar:
                video_path = os.path.join(class_path, video_name)
                npy_filename = os.path.splitext(video_name)[0] + '.npy'
                npy_path = os.path.join(out_class_path, npy_filename)
                
                if os.path.exists(npy_path):
                    skipped_count += 1
                    # Updates the right side of the progress bar silently
                    pbar.set_postfix({"Status": "Skipping", "Skipped": skipped_count, "Extracted": extracted_count})
                    continue
                
                # Show exactly which file is currently choking the CPU
                short_name = video_name[:15] + "..." if len(video_name) > 15 else video_name
                pbar.set_postfix({"Status": f"Extracting {short_name}", "Skipped": skipped_count, "Extracted": extracted_count})
                
                try:
                    spatial_data = extract_spatial_from_video(video_path, model, preprocess)
                    np.save(npy_path, spatial_data)
                    extracted_count += 1
                    # Update counts after success
                    pbar.set_postfix({"Status": "Done", "Skipped": skipped_count, "Extracted": extracted_count})
                except Exception as e:
                    # tqdm.write prints errors above the progress bar without breaking it
                    tqdm.write(f"   ⚠️ Error processing {video_name}: {e}")

if __name__ == "__main__":
    process_dataset()