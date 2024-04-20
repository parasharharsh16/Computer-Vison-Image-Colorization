data_dir = r"./data"
csv_path = f"{data_dir}/image_paths.csv"  # Path to your CSV file
data_percentage = 0.10  # e.g., 10% of the data
mode = "train"  # or 'test'
batch_size = 128 # number of images per batch
image_shape = (150, 150)  # resize all images to this shape
