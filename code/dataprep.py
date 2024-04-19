import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from constants import data_dir, csv_path, data_percentage, mode, image_shape


# Recursive function to find .jpg and .png files.
def find_images(current_path, image_list):
    # Get all entries in the current directory.
    for entry in os.listdir(current_path):
        full_path = os.path.join(current_path, entry)
        # If entry is a directory, recursively find images within.
        if os.path.isdir(full_path):
            find_images(full_path, image_list)
        elif full_path.lower().endswith((".jpg", ".png")):
            image_list.append(full_path)


# Function to write the image paths to a CSV file using pandas DataFrame.
def write_images_to_csv_with_pandas(base_path, csv_filename):
    image_list = []
    find_images(base_path, image_list)
    # Convert to DataFrame.
    df = pd.DataFrame(image_list, columns=["image_path"])
    df["test_train"] = np.where(df["image_path"].str.contains("test"), "test", "train")
    # Write to CSV using pandas to handle the file creation.
    df.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' created with {len(df)} image paths.")


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=True, resize=None):
        """
        Args:
            dataframe (DataFrame): DataFrame containing the image paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize (tuple, optional): Desired size (width, height) to resize the image.
        """
        self.data_frame = dataframe
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.data_frame)
    
    def transform_image(self, image):
        normalized_image = image / 255.0
        return normalized_image

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]

        # Load the image in color
        image_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_color is None:
            raise FileNotFoundError(
                f"The image at path {img_path} could not be loaded."
            )
        # Resize the grayscale image if resize dimensions are provided
        if self.resize:
            image_color = cv2.resize(image_color, self.resize)

        # Convert the color image to Lab color space
        image_lab = cv2.cvtColor(image_color, cv2.COLOR_BGR2Lab)

        # Load the image in grayscale
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # # Apply transformations if provided
        if self.transform:
            image_lab = self.transform_image(image_lab)
            image_gray = self.transform_image(image_gray)

        # Convert numpy arrays to tensors
        image_lab_tensor = (
            torch.from_numpy(image_lab).permute(2, 0, 1).float()
        )  # Reorder dimensions for Lab
        image_gray_tensor = (
            torch.from_numpy(image_gray).unsqueeze(0).float()
        )  # Add channel dimension for grayscale

        return image_lab_tensor, image_gray_tensor


def sample_data(file_path, data_type, sample_percentage):
    """
    Load data from a CSV file, filter by type, shuffle, and sample a percentage of it.

    Parameters:
        file_path (str): Path to the CSV file.
        data_type (str): Type of data to filter ('train' or 'test').
        sample_percentage (float): Percentage of the data to sample (e.g., 0.1 for 10%).

    Returns:
        pd.DataFrame: A DataFrame containing the sampled data.
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Shuffle the data
    shuffled_data = data.sample(frac=1, random_state=22).reset_index(drop=True)
    # Sample a percentage of the data
    sampled_data = shuffled_data.sample(frac=sample_percentage, random_state=22)

    if data_type == "train":
        filtered_data = sampled_data[sampled_data["test_train"] == data_type]
        final_data = filtered_data.sample(frac=0.8, random_state=22)
    elif data_type == "val":
        filtered_data = sampled_data[sampled_data["test_train"] == "train"]
        final_data = filtered_data.sample(frac=0.2, random_state=22)
    elif data_type == "test":
        final_data = sampled_data[sampled_data["test_train"] == data_type]
    return final_data


# if __name__ == "__main__":
#     write_images_to_csv_with_pandas(data_dir, csv_path)
#     if not os.path.exists(csv_path):
#         raise Exception("csv file with paths are not present")
#     train_df = sample_data(csv_path, "train", data_percentage)
#     val_df = sample_data(csv_path, "train", data_percentage)
#     # test_df = sample_data(csv_path, "train", data_percentage)
#     train_dataset = CustomDataset(train_df, resize=image_shape)
#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     print("ok")
    # val_dataset = CustomDataset(
    #     val_df,
    #     resize=image_shape,
    # )
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # test_dataset = CustomDataset(
    #     test_df,
    #     resize=image_shape,
    # )
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
