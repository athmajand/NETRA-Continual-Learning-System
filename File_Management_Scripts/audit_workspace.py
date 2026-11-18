import os
import cv2

# ==============================
# CONFIGURATION
# ==============================
WORKSPACE_DIR = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train"
TARGET_FRAMES = 150
# ==============================

def audit_workspace():
    print("🚀 INITIATING GRAND WORKSPACE AUDIT...")
    print(f"📂 Scanning Directory: {WORKSPACE_DIR}\n")
    
    if not os.path.exists(WORKSPACE_DIR):
        print(f"❌ ERROR: Directory not found!")
        return

    classes = [d for d in os.listdir(WORKSPACE_DIR) if os.path.isdir(os.path.join(WORKSPACE_DIR, d))]
    
    total_perfect = 0
    corrupted_files = []

    for cls in classes:
        class_path = os.path.join(WORKSPACE_DIR, cls)
        videos = [f for f in os.listdir(class_path) if f.lower().endswith('.mp4')]
        
        if not videos:
            continue
            
        print(f"🔍 Checking [{cls}] - {len(videos)} videos...")
        
        for vid in videos:
            vid_path = os.path.join(class_path, vid)
            cap = cv2.VideoCapture(vid_path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if frames == TARGET_FRAMES:
                total_perfect += 1
            else:
                corrupted_files.append((cls, vid, frames))

    print("\n" + "="*60)
    print("🏁 AUDIT COMPLETE")
    print("="*60)
    print(f"✅ Perfect 150-Frame Videos: {total_perfect}")
    
    if not corrupted_files:
        print("🎉 STATUS: FLAWLESS! Your entire workspace is mathematically perfect.")
    else:
        print(f"🚨 STATUS: FAILED! Found {len(corrupted_files)} corrupted videos:")
        for cls, vid, frames in corrupted_files:
            print(f"   ❌ [{cls}] {vid} -> Has {frames} frames! (Needs {TARGET_FRAMES})")
            
if __name__ == "__main__":
    audit_workspace()