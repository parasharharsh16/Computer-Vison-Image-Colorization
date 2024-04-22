import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import color
import torch
import cv2
from constants import data_dir, csv_path, data_percentage, mode, image_shape
import torch.nn.functional as F
from PIL import Image


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
    def __init__(self, dataframe, transform=False, size: tuple[int, int] = None):
        """
        Args:
            dataframe (DataFrame): DataFrame containing the image paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize (tuple, optional): Desired size (width, height) to resize the image.
        """
        self.data_frame = dataframe
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.data_frame)

    def load_img(self, img_path):
        out_np = np.asarray(Image.open(img_path))
        if out_np.ndim == 2:
            out_np = np.tile(out_np[:, :, None], 3)
        return out_np

    def resize_img(self, img, HW, resample=3):
        return np.asarray(
            Image.fromarray(img).resize((HW[1], HW[0]), resample=resample)
        )

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]

        # Load the image in color
        image_color = self.load_img(img_path)
        # Resize the grayscale image if resize dimensions are provided
        if self.size:
            image_color_rs = self.resize_img(image_color, HW=self.size)
        else:
            image_color_rs = image_color

        # image_lab = color.rgb2lab(image_color)
        image_lab_rs = color.rgb2lab(image_color_rs)

        # img_l = image_lab[:, :, 0]
        img_l_rs = image_lab_rs[:, :, 0]
        img_ab_rs = image_lab_rs[:, :, 1:]

        # tens_l = torch.Tensor(img_l)[None, None, :, :]
        tens_rs_l = torch.Tensor(img_l_rs)[None, :, :]
        tens_rs_ab = torch.Tensor(img_ab_rs).permute(2, 0, 1)

        tensor_img = torch.from_numpy(image_lab_rs).permute(2, 0, 1)

        return tens_rs_l, tens_rs_ab, tensor_img


def recreate_image_tensor(tens_l, out_ab, mode="bilinear"):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_l, out_ab_orig), dim=1)
    return out_lab_orig


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


# def custom_collate_fn(batch):

#     tens_l = [
#         item[0] for item in batch
#     ]  # Extract data (could be tensors of different sizes)
#     # tens_rs_l = [item[1] for item in batch]
#     # tensor_img = [item[2] for item in batch]  # Extract labels if you have them

#     return tens_l, tens_rs_l, tensor_img  # Or just return data if you don't have labels


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
