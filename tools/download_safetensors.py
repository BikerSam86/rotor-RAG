#!/usr/bin/env python3
"""
Download BitNet model.safetensors file for rotor-RAG
"""
import os
import requests
from pathlib import Path

def download_safetensors():
    """Download the BitNet model.safetensors file"""
    
    # Stable Hugging Face URL for model.safetensors
    url = "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16/resolve/main/model.safetensors?download=true"
    
    # Target directory (same as LM Studio model location)
    model_dir = Path(r"C:\Users\Samuel Howells\.lmstudio\models\microsoft\bitnet-b1.58-2B-4T-gguf")
    target_file = model_dir / "model.safetensors"
    
    print("=" * 70)
    print("DOWNLOADING BITNET MODEL.SAFETENSORS")
    print("=" * 70)
    print(f"URL: {url[:80]}...")
    print(f"Target: {target_file}")
    print()
    
    # Check if already exists
    if target_file.exists():
        file_size = target_file.stat().st_size
        print(f"✅ File already exists ({file_size:,} bytes)")
        return True
    
    # Ensure directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Starting download...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total size from headers if available
        total_size = int(response.headers.get('content-length', 0))
        if total_size:
            print(f"File size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
        
        # Download with progress
        downloaded = 0
        chunk_size = 8192  # 8KB chunks
        
        with open(target_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {downloaded:,} / {total_size:,} bytes ({percent:.1f}%)", end='')
                    else:
                        print(f"\rDownloaded: {downloaded:,} bytes", end='')
        
        print()  # New line after progress
        
        final_size = target_file.stat().st_size
        print(f"✅ Download complete!")
        print(f"   File size: {final_size:,} bytes ({final_size / (1024**3):.2f} GB)")
        print(f"   Saved to: {target_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Clean up partial file
        if target_file.exists():
            target_file.unlink()
        return False

def check_complete_model():
    """Check if we have all files needed for rotor-RAG"""
    model_dir = Path(r"C:\Users\Samuel Howells\.lmstudio\models\microsoft\bitnet-b1.58-2B-4T-gguf")
    
    required_files = [
        "ggml-model-i2_s.gguf",
        "model.safetensors", 
        "config.json"
    ]
    
    print("\n" + "=" * 70)
    print("CHECKING COMPLETE MODEL FILES")
    print("=" * 70)
    
    all_present = True
    for filename in required_files:
        filepath = model_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {filename}: {size:,} bytes")
        else:
            print(f"❌ {filename}: Missing")
            all_present = False
    
    if all_present:
        print(f"\n🚀 Complete model ready for rotor-RAG testing!")
    else:
        print(f"\n⚠️  Some files missing - model may not work")
    
    return all_present

if __name__ == "__main__":
    success = download_safetensors()
    if success:
        check_complete_model()
    print(f"\n{'='*70}")