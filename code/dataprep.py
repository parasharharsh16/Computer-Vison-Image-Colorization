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
    def __init__(
        self, csv_file, data_percentage, mode="train", transform=None, resize=None
    ):
        """
        Args:
            csv_file (string): Path to the CSV file with image paths.
            data_percentage (float): Percentage of data to sample (0-1).
            mode (string): 'train' for training data, 'test' for testing data.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize (tuple, optional): Desired size (width, height) to resize the image.
        """
        self.data_frame = pd.read_csv(csv_file)
        # Sample the data based on the given percentage
        self.data_frame = self.data_frame.sample(frac=data_percentage)
        # Filter the data based on the mode
        self.data_frame = self.data_frame[self.data_frame["test_train"] == mode]
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]

        # Read the image in color using cv2
        image_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Resize the color image if resize dimensions are provided
        if self.resize:
            image_color = cv2.resize(image_color, self.resize)

        # Convert the color image to Lab color space
        image_lab = cv2.cvtColor(image_color, cv2.COLOR_BGR2Lab)

        # Read the image in grayscale using cv2
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Resize the grayscale image if resize dimensions are provided
        if self.resize:
            image_gray = cv2.resize(image_gray, self.resize)

        # If any transformations are to be applied, apply them here
        if self.transform:
            image_lab = self.transform(image_lab)
            image_gray = self.transform(image_gray)

        # Convert numpy arrays to tensors
        image_lab_tensor = torch.from_numpy(image_lab).permute(2, 0, 1).float()
        image_gray_tensor = (
            torch.from_numpy(image_gray).unsqueeze(0).float()
        )  # Add channel dimension

        return image_lab_tensor, image_gray_tensor


if __name__ == "__main__":
    write_images_to_csv_with_pandas(data_dir, csv_path)
    if not os.path.exists(csv_path):
        raise Exception("csv file with paths are not present")
    dataset = CustomDataset(
        csv_file=csv_path,
        data_percentage=data_percentage,
        mode=mode,
        resize=image_shape,
    )
