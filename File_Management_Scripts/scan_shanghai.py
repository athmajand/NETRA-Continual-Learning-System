import os

# ==============================
# CONFIGURATION
# ==============================
TARGET_DIR = r"D:\Dataset_raw\SHANGHAI"
# ==============================

def format_size(size_in_bytes):
    """Converts bytes to a highly readable format."""
    for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

def scan_directory_tree(root_dir):
    print("============================================================")
    print(f"🔍 INITIATING DEEP RECONNAISSANCE SCAN")
    print(f"🎯 Target: {root_dir}")
    print("============================================================\n")
    
    if not os.path.exists(root_dir):
        print(f"❌ ERROR: The directory {root_dir} does not exist!")
        print("Please check the folder path and try again.")
        return

    total_files = 0
    total_size = 0
    
    # os.walk automatically goes through every single folder until the very end
    for root, dirs, files in os.walk(root_dir):
        
        # Calculate how deep we are to create the perfect visual hierarchy
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        
        # Print the current Folder
        folder_name = os.path.basename(root)
        if level == 0:
            print(f"📁 {root_dir}")
        else:
            print(f"{indent}📂 {folder_name}/")
        
        # Print all Files inside this Folder
        sub_indent = ' ' * 4 * (level + 1)
        
        # Sort files so they appear in a clean, alphabetical order
        for f in sorted(files):
            file_path = os.path.join(root, f)
            try:
                # Grab the exact size and format
                size = os.path.getsize(file_path)
                total_size += size
                total_files += 1
                
                # Extract the extension (format), fallback if there is none
                ext = os.path.splitext(f)[1].upper() or '[NO EXTENSION]'
                
                print(f"{sub_indent}📄 {f}  |  Format: {ext}  |  Size: {format_size(size)}")
            except OSError:
                print(f"{sub_indent}❌ [ACCESS DENIED] {f}")
                
    print("\n============================================================")
    print(f"🏁 SCAN COMPLETE")
    print(f"📊 Total Files Found: {total_files}")
    print(f"💽 Total Dataset Size: {format_size(total_size)}")
    print("============================================================")

if __name__ == "__main__":
    scan_directory_tree(TARGET_DIR)