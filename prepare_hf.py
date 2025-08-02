#!/usr/bin/env python3
"""
Prepare files for Hugging Face Spaces deployment
"""

import os
import shutil

def prepare_hf_deployment():
    """Prepare files for Hugging Face Spaces"""
    
    print("🚀 Preparing Hugging Face Spaces deployment...")
    
    # Create deployment directory
    hf_dir = "hf_deployment"
    if os.path.exists(hf_dir):
        shutil.rmtree(hf_dir)
    os.makedirs(hf_dir)
    
    # Copy necessary files
    files_to_copy = [
        ("app_hf.py", "app.py"),  # Rename to app.py for HF Spaces
        ("requirements_hf.txt", "requirements.txt"),
        ("README_hf.md", "README.md"),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_dir, dst))
            print(f"✅ Copied {src} -> {dst}")
        else:
            print(f"❌ File not found: {src}")
    
    # Copy models directory if it exists
    if os.path.exists("models"):
        shutil.copytree("models", os.path.join(hf_dir, "models"))
        print("✅ Copied models directory")
    else:
        print("⚠️  Models directory not found - you'll need to upload your trained model")
    
    print(f"\n🎉 Deployment files ready in '{hf_dir}' directory!")
    print("\nNext steps:")
    print("1. Go to https://huggingface.co/spaces")
    print("2. Click 'Create new Space'")
    print("3. Choose 'Streamlit' as SDK")
    print("4. Upload all files from the hf_deployment directory")
    print("5. Your app will be live in a few minutes!")

if __name__ == "__main__":
    prepare_hf_deployment()
