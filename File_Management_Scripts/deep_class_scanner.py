import os
from collections import defaultdict

# The vault to deeply scan
ROOT_DIR = r"D:\Dataset_raw"
MEDIA_EXTS = ('.mp4', '.avi', '.mkv', '.mov', '.png', '.jpg', '.jpeg')

def deep_scan():
    print(f"🚀 INITIATING DEEP SCAN OF: {ROOT_DIR}...\n")
    
    # Maps Parent Directory -> List of Sub-Directories containing media
    hierarchy = defaultdict(list)
    
    for root, dirs, files in os.walk(ROOT_DIR):
        media_files = [f for f in files if f.lower().endswith(MEDIA_EXTS)]
        if media_files:
            parent_dir = os.path.dirname(root)
            folder_name = os.path.basename(root)
            file_count = len(media_files)
            
            # Determine if this folder holds videos or image frames
            vids = sum(1 for f in media_files if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')))
            imgs = file_count - vids
            media_type = "Videos" if vids > imgs else "Images"
            
            hierarchy[parent_dir].append({
                'name': folder_name,
                'count': file_count,
                'type': media_type
            })
    
    print("📊 IDENTIFIED DATASET CLASSES AND MEDIA LOCATIONS:\n")
    
    for parent, sub_folders in hierarchy.items():
        print(f"📦 Parent Directory: {parent}")
        
        # If a parent has more than 10 sub-folders (like video_01, video_02), summarize it
        if len(sub_folders) > 10:
            total_files = sum(f['count'] for f in sub_folders)
            main_type = sub_folders[0]['type']
            print(f"   ┣━ ⚠️ Contains {len(sub_folders)} sub-folders (e.g., '{sub_folders[0]['name']}', '{sub_folders[1]['name']}')")
            print(f"   ┗━ Total Media: ~{total_files} {main_type}")
        else:
            for idx, folder in enumerate(sub_folders):
                branch = "┗━" if idx == len(sub_folders) - 1 else "┣━"
                print(f"   {branch} 📂 Folder/Class: {folder['name']} ({folder['count']} {folder['type']})")
        print("-" * 70)

if __name__ == "__main__":
    deep_scan()