import kagglehub
import os
import shutil

def download_dataset():
    print("Downloading dataset...")
  
    path = kagglehub.dataset_download("vbookshelf/rice-leaf-diseases")
    
    print("Path to dataset files:", path)
    
 
    dest_dir = "data"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    

    
    source_folder = os.path.join(path, "rice_leaf_diseases")
    if os.path.exists(source_folder):
        print(f"Moving files from {source_folder} to {dest_dir}")
        for item in os.listdir(source_folder):
            s = os.path.join(source_folder, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print("Dataset successfully moved to 'data' folder.")
    else:
       
        print(f"Moving files from {path} to {dest_dir}")
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print("Dataset successfully moved to 'data' folder.")

if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have kaggle credentials configured or download manually from:")
        print("https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases")
