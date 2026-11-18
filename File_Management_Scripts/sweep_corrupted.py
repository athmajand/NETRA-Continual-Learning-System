import os
import numpy as np
from glob import glob

# ==============================
# CONFIGURATION
# ==============================
SPATIAL_DIR = r"D:\Threat_Detection_Workspace\02_Feature_Extraction\Spatial_Features_Granular"
# ==============================

def sweep_for_corruption():
    print("🧹 INITIATING CORRUPTION SWEEPER...")
    print(f"Scanning {SPATIAL_DIR} for broken .npy files from the power outage...\n")
    
    search_path = os.path.join(SPATIAL_DIR, "**", "*.npy")
    files = glob(search_path, recursive=True)
    
    corrupted_count = 0
    
    for file in files:
        try:
            # Try to load the array. If it's corrupted, this will crash and trigger the except block.
            np.load(file)
        except Exception as e:
            print(f"   🚨 CORRUPTED FILE DETECTED: {os.path.basename(file)}")
            print("      -> Deleting file for regeneration...")
            os.remove(file)
            corrupted_count += 1

    print("\n" + "="*60)
    if corrupted_count == 0:
        print("✅ NO CORRUPTION FOUND! Your files survived the crash perfectly.")
    else:
        print(f"🧹 SWEEP COMPLETE! Successfully deleted {corrupted_count} corrupted files.")
    print("="*60)

if __name__ == "__main__":
    sweep_for_corruption()