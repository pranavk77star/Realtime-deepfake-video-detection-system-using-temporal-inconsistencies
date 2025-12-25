import os
import shutil

# Change this to your dataset folder
dataset_folder = r"datasets\frames\Celeb-real"

# Loop through all subfolders
for folder in os.listdir(dataset_folder):
    folder_path = os.path.join(dataset_folder, folder)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            src = os.path.join(folder_path, f)
            dst = os.path.join(dataset_folder, f)
            shutil.move(src, dst)  # move file to main folder
        os.rmdir(folder_path)  # remove empty subfolder

print(f"✅ Flattened {dataset_folder}")
