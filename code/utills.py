import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
#import tqdm 
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt
import numpy as np
class GrayscaleToColorCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(GrayscaleToColorCNN, self).__init__()

        # Encoder (Grayscale to latent space)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        # Decoder (latent space to ab channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output range [-1, 1] for ab channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ClassBalancedLoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(ClassBalancedLoss, self).__init__()
        self.weights = weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.reduction = reduction

    def forward(self, output, target):
        # Calculate per-pixel loss
        loss = F.mse_loss(output, target, reduction=self.reduction)

        # Apply class rebalancing weights
        if self.weights is not None:
            loss = loss * self.weights

        return loss

def create_color_weights(quantized_ab_space, image_net_data, lambda_=0.5, sigma=5):

    # Estimate empirical probability distribution
    p = F.avg_pool2d(image_net_data, kernel_size=quantized_ab_space.shape[1:])

    # Smooth distribution with Gaussian kernel
    smoothed_p = F.conv2d(p.unsqueeze(1), torch.ones((1, 1, sigma, sigma)), padding=sigma // 2)

    # Mix with uniform distribution
    uniform_dist = torch.ones_like(smoothed_p)
    mixed_dist = (1 - lambda_) * smoothed_p + lambda_ * uniform_dist

    # Calculate weights and normalize
    weights = 1.0 / (mixed_dist + 1e-8)
    weights = weights.view(-1)  # Flatten for efficiency

    return weights

def combine_lab(l_channel, ab_channels):
    # Ensure that both inputs are PyTorch tensors
    l_channel = l_channel.float()
    ab_channels = ab_channels.float()

    # Rescale L channel to [0, 100] range
    l_channel = (l_channel * 100.0).clamp(0, 100)

    # Rescale AB channels to [-128, 128] range
    ab_channels = (ab_channels * 128.0).clamp(-128, 128)

    # Combine L, AB channels to create LAB image
    lab_image = torch.cat((l_channel, ab_channels), dim=1)

    return lab_image

def train_model(model, train_loader, optimizer, criterion, num_epochs,device):

    model.train()  # Set model to training mode

    log_interval = len(train_loader)
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (targets,images) in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            output_with_L = combine_lab(images,outputs)
            # Calculate loss with class rebalancing
            loss = criterion(output_with_L, targets)

            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}")
            # Print training progress (optional)
            # if (i + 1) % log_interval == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    return model

def predict(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    output_images = []
    with torch.no_grad():
        for i, (targets,images) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(images)
            output_with_L = torch.cat((images,outputs), dim=1)
            output_images.append([images[0],targets[0],output_with_L[0]])

    return output_images
def denormalize_lab_image(lab_image_normalized):
    lab_image_denormalized = np.zeros_like(lab_image_normalized, dtype=np.uint8)

    # Scale L from [0, 100] to [0, 255]
    lab_image_denormalized[..., 0] = np.clip((lab_image_normalized[..., 0] * 2.55), 0, 255).astype(np.uint8)
    
    # Scale and shift a* and b* from [-128, 127] to [0, 255]
    lab_image_denormalized[..., 1] = np.clip((lab_image_normalized[..., 1] * 1.27 + 128), 0, 255).astype(np.uint8)
    lab_image_denormalized[..., 2] = np.clip((lab_image_normalized[..., 2] * 1.27 + 128), 0, 255).astype(np.uint8)

    return lab_image_denormalized
def plot_images(inputimg, targetimg, predictedimg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a single row with three subplots

    # Convert input grayscale image to RGB
    input_rgb = cv2.cvtColor(np.transpose(inputimg.detach().cpu().numpy(), (1, 2, 0)), cv2.COLOR_GRAY2RGB)
    # Plot input image
    axes[0].imshow(input_rgb, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Plot target image
    target_rgb = cv2.cvtColor(denormalize_lab_image(np.transpose(targetimg.detach().cpu().numpy(), (1, 2, 0))), cv2.COLOR_LAB2RGB)
    #target_rgb = cv2.cvtColor((targetimg.detach().cpu().numpy()).permute(1, 2, 0), cv2.COLOR_LAB2RGB)
    
    axes[1].imshow(target_rgb)
    axes[1].set_title('Target Image')
    axes[1].axis('off')

    # Plot predicted image
    predicted_rgb = cv2.cvtColor(denormalize_lab_image(np.transpose(predictedimg.detach().cpu().numpy(), (1, 2, 0))), cv2.COLOR_LAB2RGB)
    axes[2].imshow(predicted_rgb)
    axes[2].set_title('Predicted Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("TestImage.jpg")
    plt.show()