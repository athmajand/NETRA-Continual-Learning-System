import os
import glob
import yt_dlp
import time

# ==============================
# CONFIGURATION
# ==============================
BASE_OUTPUT_DIR = r"D:\Dataset_raw\YouTube_Auto_Factory"
ARCHIVE_FILE = os.path.join(BASE_OUTPUT_DIR, "master_download_ledger.txt")
TARGET_VIDEOS_PER_CLASS = 250  

# 🌟 MULTI-QUERY MATRIX: Bypassing YouTube's single-phrase limits 🌟
QUERIES = [
    "caught on CCTV stone pelting street",
    "throwing rocks at cars dashcam",
    "vandals throwing stones at windows CCTV",
    "mob pelting stones caught on camera",
    "youths throwing rocks at vehicles"
]
# ==============================

def autonomous_harvest():
    class_name = "Stone_Pelting"
    print(f"🤖 INITIATING STEALTH MULTI-QUERY FACTORY...")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    class_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for query in QUERIES:
        # Count what we have before starting the next query
        existing_mp4s = glob.glob(os.path.join(class_dir, "*.mp4"))
        initial_count = len(existing_mp4s)
        needed_count = TARGET_VIDEOS_PER_CLASS - initial_count
        
        if needed_count <= 0:
            print(f"\n✅ Goal completely met! Found {initial_count} videos.")
            break

        print(f"\n" + "="*60)
        print(f"🎯 SEARCHING: [{query}]")
        print(f"📊 Current Folder Count: {initial_count} | Needed: {needed_count}")
        
        # Search 75 results per specific query instead of 500 on one
        deep_search_query = f"ytsearch75:{query}" 

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(class_dir, '%(title)s.%(ext)s'), 
            'merge_output_format': 'mp4',
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': False,
            
            'download_archive': ARCHIVE_FILE,
            'match_filter': yt_dlp.utils.match_filter_func("duration <= 900"),
            'max_downloads': needed_count,
            'cookiefile': r"D:\File_Management_Scripts\youtube.com_cookies.txt",
            
            # 🌟 STEALTH MODE: Adds a 2 to 5 second random pause between downloads to prevent DNS bans
            'sleep_interval': 2,
            'max_sleep_interval': 5,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([deep_search_query])
        except Exception as e:
            print(f"⚠️ Search interrupted: {e}")
            
        # Give the network a quick 5-second breather before hitting the next query
        time.sleep(5)

    final_mp4s = glob.glob(os.path.join(class_dir, "*.mp4"))
    print("\n" + "="*60)
    print(f"🏁 FINAL HARVEST COMPLETE: {len(final_mp4s)} Total Videos in Folder.")
    print("="*60)

if __name__ == "__main__":
    autonomous_harvest()