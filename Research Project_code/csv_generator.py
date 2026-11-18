import os
import csv

# ==============================
# CONFIGURATION
# ==============================
# 🎯 The source directory containing your folders/classes of data
TARGET_DIR = r"C:\Users\acer\Desktop\Helmet Detection using YOLO"

# 📝 The path where your clean Hugging Face metadata.csv file will be generated
OUTPUT_CSV = r"C:\Users\acer\Desktop\Athmajan\metadata.csv"
# ==============================

def format_size(size_in_bytes):
    """Converts bytes to a highly readable format."""
    for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

def build_hf_metadata_csv(root_dir, csv_output_path):
    print("============================================================")
    print("🔍 INITIATING DATASET ARCHITECTURE SCAN")
    print(f"🎯 Target Directory: {root_dir}")
    print(f"📝 Output Metadata CSV: {csv_output_path}")
    print("============================================================\n")
    
    if not os.path.exists(root_dir):
        print(f"❌ ERROR: The directory {root_dir} does not exist!")
        return

    # Ensure output directory wrapper exists
    output_dir = os.path.dirname(csv_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_files = 0
    total_size = 0
    
    # Define columns required by Hugging Face (file_name must be exact lowercase)
    csv_headers = ["file_name", "label", "file_format", "file_size_readable"]

    try:
        with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            # Write the header row
            writer.writerow(csv_headers)
            
            # os.walk goes deep down through all class subdirectories
            for root, dirs, files in os.walk(root_dir):
                for f in sorted(files):
                    # Skip tracking system hidden assets or metadata files themselves if run inside
                    if f.lower() in ['thumbs.db', '.ds_store', 'metadata.csv']:
                        continue
                        
                    full_file_path = os.path.join(root, f)
                    
                    try:
                        # 1. Compute Relative Path for Hugging Face indexing
                        # Example: 'train/helmet/img1.jpg' instead of full local C:\... paths
                        rel_path = os.path.relpath(full_file_path, root_dir)
                        # Replace Windows backslashes with forward slashes for universal compatibility
                        rel_path = rel_path.replace(os.sep, '/')
                        
                        # 2. Extract Label (The direct parent folder name acts as the category class)
                        label = os.path.basename(root)
                        if root == root_dir:
                            label = "root_unclassified" # Fallback if files sit right in the main folder
                        
                        # 3. File specifics
                        ext = os.path.splitext(f)[1].lower() or 'none'
                        size_bytes = os.path.getsize(full_file_path)
                        
                        total_size += size_bytes
                        total_files += 1
                        
                        # Write structured row data into CSV
                        writer.writerow([rel_path, label, ext.upper(), format_size(size_bytes)])
                        
                    except OSError:
                        print(f"⚠️ [ACCESS DENIED / READ ERROR] Skipping file: {f}")

        print("============================================================")
        print("🏁 METADATA EXPORT COMPLETE")
        print(f"📊 Total Dataset Files Indexed: {total_files}")
        print(f"💽 Total Aggregated Size: {format_size(total_size)}")
        print(f"🚀 CSV generated safely at: {csv_output_path}")
        print("============================================================")

    except Exception as e:
        print(f"❌ Critical Error writing CSV output file: {str(e)}")

if __name__ == "__main__":
    build_hf_metadata_csv(TARGET_DIR, OUTPUT_CSV)