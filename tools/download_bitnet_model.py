#!/usr/bin/env python3
"""
Download the official BitNet-b1.58-2B-4T model from Hugging Face
"""
import os
from huggingface_hub import snapshot_download

def download_bitnet_model():
    """Download BitNet 2B model to a local directory"""
    
    model_repo = "microsoft/BitNet-b1.58-2B-4T"
    local_dir = r"C:\Users\Samuel Howells\Google Drive\GitHub TriStar Personal\BitNet-2B-model"
    
    print("=" * 70)
    print("DOWNLOADING BITNET 2B MODEL FROM HUGGING FACE")
    print("=" * 70)
    print(f"Repository: {model_repo}")
    print(f"Local directory: {local_dir}")
    print()
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        print("Starting download...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Copy files instead of symlinks on Windows
        )
        print()
        print("✅ Download complete!")
        print(f"Model saved to: {local_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), local_dir)
                size = os.path.getsize(os.path.join(root, file))
                print(f"  {rel_path} ({size:,} bytes)")
                
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = download_bitnet_model()
    if success:
        print("\n🚀 Ready to test with real BitNet model!")
    else:
        print("\n💥 Download failed. Check internet connection and try again.")