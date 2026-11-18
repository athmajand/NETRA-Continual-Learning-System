import os
import cv2

# ================= CONFIGURATION =================
BASE_DIR = r"D:\Dataset_raw\Fence_Climbing"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")
# =================================================

def scan_folder(folder_path, split_name):
    print(f"\n🔍 Scanning '{split_name}' directory...")
    print(f"   Path: {folder_path}")
    
    if not os.path.exists(folder_path):
        print("   ❌ ERROR: Folder does not exist!")
        return
        
    # Grab all image files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print("   ⚠️ WARNING: No images found directly in this folder. Are they inside an 'images' subfolder?")
        
        # Check if there is an 'images' subfolder
        if os.path.exists(os.path.join(folder_path, "images")):
             print("   💡 UPDATE: Found an 'images' subfolder! Please update your paths to point inside it.")
        return
        
    # Sort files to check naming sequence
    files.sort()
    count = len(files)
    print(f"   📁 Total Images Found: {count}")
    print(f"   📄 First file: {files[0]}")
    print(f"   📄 Last file: {files[-1]}")
    
    # Read the first image to get the exact resolution for OpenCV
    sample_path = os.path.join(folder_path, files[0])
    img = cv2.imread(sample_path)
    
    if img is not None:
        height, width, channels = img.shape
        print(f"   📐 Image Resolution: {width} x {height} pixels")
    else:
        print("   ❌ ERROR: Could not read the image file. It might be corrupted.")

def main():
    print("🚀 INITIATING DATASET DIAGNOSTIC SCANNER...")
    scan_folder(TRAIN_DIR, "Train")
    scan_folder(TEST_DIR, "Test")
    print("\n" + "="*60)
    print("🏁 SCAN COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()