import torch
import torch.nn as nn
import torch.nn.functional as F
#import tqdm 
from tqdm import tqdm # type: ignore
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
            output_with_L = torch.cat((images,outputs), dim=1)
            # Calculate loss with class rebalancing
            loss = criterion(output_with_L, targets)

            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}")
            # Print training progress (optional)
            if (i + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


