import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

source_dir = "/bigdata/akshay/datacollection/error_detection/dataset/features/x3d_1"
destination_dir = (
    "/bigdata/akshay/datacollection/error_detection/dataset/features/x3d_pca_nc64"
)
count = 0
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


def apply_pca(file_path):
    data = np.load(file_path)
    data = data.squeeze(0)
    # reshaped_data = data.reshape(2048, 512)
    reshaped_data = data.reshape(192, 1024) # for x3d
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(reshaped_data)
    pca = PCA(n_components=64)
    pca_data = pca.fit_transform(normalized_data)
    return pca_data


for subdir, _, files in os.walk(source_dir):
    # Replace the source directory path in subdir with the destination directory path
    dest_subdir = subdir.replace(source_dir, destination_dir)

    # Check if the corresponding subdir already exists in the destination directory
    if os.path.exists(dest_subdir):
        print(f"{dest_subdir} already exists. Skipping...")
        continue  # Skip this iteration and move to the next subdir

    print(subdir)
    for file in tqdm(files):
        if file.endswith(".npy"):
            # Get the full path of the current file
            full_file_path = os.path.join(subdir, file)

            # Apply PCA to the file data
            transformed_data = apply_pca(full_file_path)

            # Create the equivalent path in the destination directory
            dest_subdir = subdir.replace(source_dir, destination_dir)
            if not os.path.exists(dest_subdir):
                os.makedirs(dest_subdir)

            dest_file_path = os.path.join(dest_subdir, file)

            # Save the transformed data in the new location
            np.save(dest_file_path, transformed_data)
    count += 1
    print(count)
    print()


print("PCA transformation complete!")
