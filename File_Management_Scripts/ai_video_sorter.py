import os
import cv2
import torch
import shutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
# Point this to the folder containing your leftover Shanghai clips
LEFTOVER_DIR = r"C:\Users\acer\Desktop\Manual Dataset\shanhai"
# Point this to where you want the AI to sort them
OUTPUT_DIR = r"D:\Dataset_raw\AI_Sorted_Clips"

# These are the exact text descriptions the AI will use to categorize your videos
CATEGORIES = [
    "people fighting or brawling",
    "a person running or chasing someone",
    "a person riding a bicycle or skateboard",
    "a person dropping a bag or luggage",
    "a crowd of people panicking",
    "normal people walking peacefully"
]
# ==============================

def get_middle_frame(video_path):
    """Extracts the middle frame of a video for the AI to analyze."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR (OpenCV) to RGB (PIL)
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def run_ai_sorter():
    print("🤖 INITIATING AI CLIP SORTER...")
    
    # Load the AI Model (Downloads automatically the first time)
    print("   -> Loading Vision-Language Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create the output folders
    for cat in CATEGORIES:
        # Create safe folder names (replace spaces with underscores)
        folder_name = cat.replace(" ", "_")
        os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)
        
    videos = [f for f in os.listdir(LEFTOVER_DIR) if f.endswith('.mp4')]
    
    if not videos:
        print("❌ No videos found in the leftover directory!")
        return

    print(f"\n🔍 Analyzing and sorting {len(videos)} videos on device: {device}...\n")
    
    for vid in tqdm(videos, desc="AI Sorting"):
        vid_path = os.path.join(LEFTOVER_DIR, vid)
        
        # 1. Get the middle frame
        frame = get_middle_frame(vid_path)
        if frame is None:
            continue
            
        # 2. Ask the AI to predict the category
        inputs = processor(text=CATEGORIES, images=frame, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity score
            probs = logits_per_image.softmax(dim=1)      # convert to probabilities
            
        # 3. Find the winning category
        best_match_idx = probs.argmax().item()
        best_category = CATEGORIES[best_match_idx]
        
        # 4. Move the file to the AI's predicted folder
        safe_folder_name = best_category.replace(" ", "_")
        dest_path = os.path.join(OUTPUT_DIR, safe_folder_name, vid)
        shutil.move(vid_path, dest_path)

    print("\n" + "="*60)
    print(f"✅ AI SORTING COMPLETE! Check your folders at: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    run_ai_sorter()