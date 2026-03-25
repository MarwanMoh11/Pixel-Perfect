import os
import zipfile
import subprocess
import sys

def download_kaggle_dataset():
    dataset_name = "ebrahimelgazar/pixel-art"
    download_dir = "data/raw"
    
    # Ensure the directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Downloading '{dataset_name}' into '{download_dir}'...")
    try:
        # Run the kaggle CLI tool using the active python environment
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", download_dir], 
            check=True
        )
    except subprocess.CalledProcessError:
        print("\nERROR: Failed to download dataset.")
        print("Please ensure your kaggle.json API key is placed in ~/.kaggle/kaggle.json")
        print("and has the correct permissions (chmod 600 ~/.kaggle/kaggle.json)")
        sys.exit(1)
    except FileNotFoundError:
        print("\nERROR: 'kaggle' command not found. Did you activate the venv and install requirements?")
        sys.exit(1)
        
    zip_path = os.path.join(download_dir, "pixel-art.zip")
    
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
            
        print("Dataset extracted successfully.")
        
        # Clean up the zip file to save space
        os.remove(zip_path)
        print("Cleaned up zip file.")
    else:
        print("Download failed or zip file not found.")

if __name__ == "__main__":
    download_kaggle_dataset()
