import os
import math

# ==============================
# CONFIGURATION
# ==============================
TARGET_DIR = r"D:\Dataset_raw\SHANGHAI\SHANGHAI_TRAIN"
# ==============================

def format_size(size_bytes):
    """Converts raw bytes into a human-readable format (KB, MB, GB)."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def scan_directory():
    print("🔍 INITIATING DEEP DIRECTORY SCAN...")
    print(f"Target: {TARGET_DIR}\n")

    if not os.path.exists(TARGET_DIR):
        print(f"❌ ERROR: The directory {TARGET_DIR} does not exist.")
        print("Please check if the extraction is complete and the folder name is correct.")
        return

    total_files = 0
    total_size_bytes = 0

    # Print Table Header
    print("=" * 100)
    print(f"{'FILE NAME / RELATIVE PATH':<65} | {'FORMAT':<10} | {'SIZE':<15}")
    print("=" * 100)

    # os.walk recursively hunts down every file in every subfolder
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                # Get File Size
                size_b = os.path.getsize(file_path)
                total_size_bytes += size_b
                total_files += 1

                # Get File Format (Extension)
                _, ext = os.path.splitext(file)
                ext = ext.upper().replace(".", "") if ext else "UNKNOWN"

                # Get Relative Path (to show exactly which subfolder it is in)
                rel_path = os.path.relpath(file_path, TARGET_DIR)
                
                # Format the path so it fits perfectly in the terminal column
                display_path = rel_path if len(rel_path) <= 63 else "..." + rel_path[-60:]

                # Print the exact file data
                print(f"{display_path:<65} | {ext:<10} | {format_size(size_b):<15}")

            except Exception as e:
                print(f"⚠️ ERROR reading file: {file} - {e}")

    # Print Summary Footer
    print("=" * 100)
    print(f"🏁 SCAN COMPLETE")
    print(f"📊 Total Files Found : {total_files}")
    print(f"💾 Total Storage Used: {format_size(total_size_bytes)}")
    print("=" * 100)

if __name__ == "__main__":
    scan_directory()