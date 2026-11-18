import os

# The two most likely locations based on your previous paths
TARGET_DIRS = [
    r"D:\Dataset_raw\Abnormal\Abnormal\Anomalous classes",
    r"D:\Dataset_raw"
]

def scan_for_classes():
    print("🚀 SCANNING FOR NEW THREAT CLASSES...\n")
    
    for directory in TARGET_DIRS:
        if os.path.exists(directory):
            print(f"📂 Searching in: {directory}")
            
            # Get all sub-folders (which represent your classes)
            folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
            
            if folders:
                print(f"✅ Found {len(folders)} class folders:")
                for folder in sorted(folders):
                    print(f"   - {folder}")
            else:
                print("   ⚠️ No folders found inside this directory.")
            print("-" * 60)
        else:
            print(f"❌ Directory not found: {directory}\n" + "-" * 60)

if __name__ == "__main__":
    scan_for_classes()