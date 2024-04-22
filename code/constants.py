data_dir = r"./data"
csv_path = f"{data_dir}/image_paths.csv"  # Path to your CSV file
data_percentage = 0.01  # e.g., 10% of the data
mode = "train"  # or 'test'
batch_size = 64  # number of images per batch
image_shape = (256, 256)  # resize all images to this shape

# Hyperparams
num_epochs = 5
learning_rate = 0.001
