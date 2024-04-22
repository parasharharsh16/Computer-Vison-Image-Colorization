from utills import (
    train_model,
    predict,
    plot_images,
    calculate_auc_accuracy_plot_roc,
    # ClassBalancedLoss,
)
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from constants import (
    data_dir,
    csv_path,
    data_percentage,
    image_shape,
    batch_size,
    num_epochs,
    learning_rate,
)
from dataprep import (
    sample_data,
    write_images_to_csv_with_pandas,
    CustomDataset,
)
from eccv16 import eccv16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)


# data load
write_images_to_csv_with_pandas(data_dir, csv_path)
if not os.path.exists(csv_path):
    raise Exception("csv file with paths are not present")
train_df = sample_data(csv_path, "train", data_percentage)
# val_df = sample_data(csv_path, "val", data_percentage)
test_df = sample_data(csv_path, "test", data_percentage)
train_dataset = CustomDataset(train_df, size=image_shape)
test_dataset = CustomDataset(
    test_df,
    size=image_shape,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
print("Data Loaded")


# Model Definition
model = eccv16(pretrained=False)
model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ClassBalancedLoss with pre-calculated weights
criterion = torch.nn.MSELoss()
# criterion = ClassBalancedLoss()

# Train the model
# model = train_model(
#     model, train_loader, optimizer, batch_size, num_epochs, device, criterion
# )
model.load_state_dict(torch.load("model/model.pth"))
print_output, accuracy_lst, auc_lst = predict(model, test_loader, device)

plot_images(*print_output[0])
# plot_images(*test_output[1])

# USe of Evaluation Function
# auc_score, accuracy = calculate_auc_accuracy_plot_roc(target_images, output_images)
print("AUC:", np.max(auc_lst))
print("Accuracy:", np.max(accuracy_lst))
