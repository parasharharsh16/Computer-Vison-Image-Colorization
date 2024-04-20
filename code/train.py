from utills import train_model,predict,plot_images, ClassBalancedLoss, create_color_weights, GrayscaleToColorCNN
import torch
import os
from torch.utils.data import DataLoader
from constants import data_dir, csv_path, data_percentage, mode, image_shape,batch_size
from dataprep import (sample_data,
                      write_images_to_csv_with_pandas,
                      CustomDataset)
from base_color import *
from eccv16 import eccv16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
num_epochs = 5
learning_rate = 0.01
write_images_to_csv_with_pandas(data_dir, csv_path)
if not os.path.exists(csv_path):
    raise Exception("csv file with paths are not present")
train_df = sample_data(csv_path, "train", data_percentage)
val_df = sample_data(csv_path, "val", data_percentage)
test_df = sample_data(csv_path, "test", data_percentage)
# test_df = sample_data(csv_path, "train", data_percentage)
train_dataset = CustomDataset(train_df, resize=image_shape)
test_dataset = CustomDataset(test_df, resize=image_shape)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader =  DataLoader(test_dataset,batch_size=1,shuffle=True)
print("ok")


# Model Definition
model = eccv16(pretrained=False)
model.to(device)

# Calculate class weights (consider adjusting hyperparameters)
# quantized_ab_space = torch.zeros((11, 11, 2))  # Example quantized ab space (adjust based on your setup)
# class_weights = create_color_weights(quantized_ab_space, next(iter(train_loader))[0], lambda_=0.5, sigma=5)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ClassBalancedLoss with pre-calculated weights
#write criterion using torch

# criterion = torch.nn.MSELoss()
# Example usage
# Assuming 'preds' is the output from your network, 'true' is the soft-encoded ground truth,
# and 'weights' is the class weighting factor obtained from the training data analysis.
# preds, true, and weights should all be torch Tensors.



# Train the model
model = train_model(model, train_loader, optimizer, batch_size,num_epochs,device)
#model.load_state_dict(torch.load("model/model.pth"))
test_output = predict(model,test_loader,device)

plot_images(*test_output[0])
#plot_images(*test_output[1])
