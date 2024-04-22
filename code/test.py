from dataprep import (
    sample_data,
    write_images_to_csv_with_pandas,
    CustomDataset,
)
from constants import data_dir, csv_path, data_percentage, mode, image_shape, batch_size
import torch
from torch.utils.data import DataLoader
import os


write_images_to_csv_with_pandas(data_dir, csv_path)
if not os.path.exists(csv_path):
    raise Exception("csv file with paths are not present")
train_df = sample_data(csv_path, "train", data_percentage)
val_df = sample_data(csv_path, "val", data_percentage)
test_df = sample_data(csv_path, "test", data_percentage)
# test_df = sample_data(csv_path, "train", data_percentage)
train_dataset = CustomDataset(train_df, size=image_shape)
test_dataset = CustomDataset(
    test_df,
    size=image_shape,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def calculate_batch_class_weights(target):
    # Assuming target is a batch of ab channel images with shape (batch_size, 2, height, width)
    image_ab_batches = target

    # Splitting a and b channels while preserving batch and channel dimensions
    a = image_ab_batches[:, 0, :, :].unsqueeze(1)
    b = image_ab_batches[:, 1, :, :].unsqueeze(1)

    # Define bins
    bins = torch.arange(-128, 128, 8).to(a.device)

    # Digitize a and b channels into bins
    a_bins = torch.bucketize(a, bins, right=True)
    b_bins = torch.bucketize(b, bins, right=True)

    # Combine the a and b bins to get a single label for each pixel
    y_true = a_bins * 256 + b_bins

    # Flatten y_true along each batch, preserving the batch dimension
    y_true_flat = y_true.view(y_true.shape[0], -1)  # Flattening only spatial dimensions

    # Initialize a list to store the class weights for each batch
    batch_class_weights = []

    for i in range(y_true_flat.shape[0]):
        # Calculate max value for creating bincount tensor for the current batch
        max_val = y_true_flat[i].max().item() + 1

        # Count the number of pixels of each class in the current batch image
        class_counts = torch.bincount(y_true_flat[i], minlength=max_val)

        # Compute the class weights for the current batch image
        class_weights = 1.0 / (class_counts.float() + 1e-5)

        # Append the calculated class weights to the list
        batch_class_weights.append(class_weights)

    # Optionally convert list of tensors to a single tensor if uniform length is expected
    # This depends on whether you expect to use varying numbers of classes across different images
    # batch_class_weights = torch.stack(batch_class_weights)

    return batch_class_weights


for x, y, z in train_loader:
    print(x.shape)
    print(y.shape)
    print(z.shape)
    weights = calculate_batch_class_weights(y)
    break
