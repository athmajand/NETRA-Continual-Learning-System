from huggingface_hub import HfApi
import os

# ================= CONFIGURATION =================
# The folder containing all your final codes, models, and docs
FOLDER_TO_UPLOAD = r"D:\Research Project_code"

# Your exact Hugging Face username and the repository name you want
REPO_ID = "athmajandilip/NETRA-Continual-Learning-System"
# =================================================

def push_to_hf():
    api = HfApi()
    
    print(f"🚀 Creating repository: {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    
    print(f"⬆️ Uploading all files from {FOLDER_TO_UPLOAD}...")
    api.upload_folder(
        folder_path=FOLDER_TO_UPLOAD,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Initial commit: NETRA EWC Continual Learning Framework & UI"
    )
    
    print(f"✅ Upload Complete! Check your profile: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    push_to_hf()