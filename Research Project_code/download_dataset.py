from huggingface_hub import snapshot_download

print("⏳ Initiating Hugging Face Download...")

snapshot_download(
    repo_id="DiricziZsolt/Latch-Video-Package-Theft",
    repo_type="dataset",
    local_dir=r"D:\Research Project_code\Raw_Data\Package_Theft",
    local_dir_use_symlinks=False
)

print("✅ Download Complete! Check your Raw_Data folder.")