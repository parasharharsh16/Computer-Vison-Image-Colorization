import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# import tqdm
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from dataprep import recreate_image_tensor


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
            nn.MaxPool2d(2, stride=2),
        )

        # Decoder (latent space to ab channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # Output range [-1, 1] for ab channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ClassBalancedLoss(nn.Module):
    def __init__(self, weights, reduction="mean"):
        super(ClassBalancedLoss, self).__init__()
        self.weights = weights.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
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
    smoothed_p = F.conv2d(
        p.unsqueeze(1), torch.ones((1, 1, sigma, sigma)), padding=sigma // 2
    )

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

    return denormalize_lab_image(lab_image)


def weighted_multinomial_cross_entropy(output, target, weights):
    """
    Calculate the weighted multinomial cross-entropy loss.

    Args:
        output (torch.Tensor): The predicted probabilities for each class, shape (batch_size, H, W, Q)
        target (torch.Tensor): The ground truth probabilities, shape (batch_size, H, W, Q)
        weights (torch.Tensor): Weight for each class in the quantized ab space, shape (Q,)

    Returns:
        torch.Tensor: The weighted cross-entropy loss.
    """
    # Log of probabilities to prevent numerical instability
    log_probs = torch.log(output.clamp(min=1e-5))

    # Multiply log probabilities by the target probabilities
    weighted_log_probs = target * log_probs

    # Sum across the class dimension (Q)
    class_sum = weighted_log_probs.sum(dim=-1)

    # Find the maximum probability index for each pixel to determine the weight
    max_indices = target.argmax(dim=-1)
    pixel_weights = weights[max_indices]

    # Apply the pixel weights
    weighted_loss = class_sum * pixel_weights

    # Average over all pixels and batch size
    return -weighted_loss.mean()


def multinomial_cross_entropy_loss(Z_hat, Z):
    """
    Standard multinomial cross-entropy loss.
    :param Z_hat: Predicted probability distributions (N, H, W, Q)
    :param Z: Target distributions (N, H, W, Q)
    :return: Cross-entropy loss
    """
    # Flatten the tensors to simplify the loss calculation
    Z_hat_flat = Z_hat.view(-1, Z_hat.shape[-1])  # (N*H*W, Q)
    Z_flat = Z.view(-1, Z.shape[-id])  # (N*H*W, Q)

    # Compute log probabilities
    log_prob = torch.log(Z_hat_flat + 1e-8)  # add a small number to prevent log(0)

    # Compute the cross-entropy
    loss = -torch.sum(
        Z_flat * log_prob, dim=1
    )  # element-wise multiplication and sum over Q

    # Return the mean of the loss across all pixels
    return torch.mean(loss)


def train_model(
    model, train_loader, optimizer, batch_size, num_epochs, device, criterion
):

    model.train()  # Set model to training mode

    log_interval = len(train_loader)
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (tens_rs_l, tens_rs_ab, tensor_rs_img) in progress_bar:
            tens_rs_l = tens_rs_l.to(device).float()
            tens_rs_ab = tens_rs_ab.to(device).float()
            tensor_rs_img = tensor_rs_img.to(device).float()

            # Forward pass
            outputs = model(tens_rs_l)
            output_with_L = torch.cat((tens_rs_l, outputs), dim=1)
            # Calculate loss with class rebalancing
            # output_lab_img = recreate_image_tensor(tens_rs_l, outputs, mode="bilinear")
            # target_ab_channels = targets[:, 1:, :, :]
            # loss = criterion(outputs, tens_rs_ab)
            # H, W, Q = 150, 150, 2
            # weights = torch.rand(
            #     Q, dtype=torch.float32
            # )  # Random weights for each class
            # weights = weights.to(device)

            loss = criterion(output_with_L, tensor_rs_img)
            # loss = multinomial_cross_entropy_loss(output_with_L, tensor_rs_img)

            # Backward pass and update weights
            optimizer.zero_grad()
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()
            progress_bar.set_description(
                f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}"
            )
            # Print training progress (optional)
            # if (i + 1) % log_interval == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), "model/model.pth")
    return model


def predict(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    output_images = []
    with torch.no_grad():
        for i, (tens_rs_l, tens_rs_ab, tensor_rs_img) in enumerate(test_loader):
            tens_rs_l = tens_rs_l.to(device).float()
            tens_rs_ab = tens_rs_ab.to(device).float()
            tensor_rs_img = tensor_rs_img.to(device).float()
            # Forward pass
            outputs = model(tens_rs_l)
            output_with_L = torch.cat((tens_rs_l, outputs), dim=1)
            # output_with_L = torch.zeros_like(images)
            output_images.append([tens_rs_l[0], tensor_rs_img[0], output_with_L[0]])
            break
    return output_images


def denormalize_lab_image(lab_image_normalized):
    dtype = np.uint8
    lab_image_denormalized = np.zeros_like(lab_image_normalized, dtype=dtype)

    # Scale L from [0, 100] to [0, 255]
    lab_image_denormalized[..., 0] = np.clip(
        (lab_image_normalized[..., 0] * 2.55), 0, 255
    ).astype(dtype)

    # Scale and shift a* and b* from [-128, 127] to [0, 255]
    lab_image_denormalized[..., 1] = np.clip(
        (lab_image_normalized[..., 1] * 1.27 + 128), 0, 255
    ).astype(dtype)
    lab_image_denormalized[..., 2] = np.clip(
        (lab_image_normalized[..., 2] * 1.27 + 128), 0, 255
    ).astype(dtype)

    return lab_image_denormalized


def plot_images(inputimg, targetimg, predictedimg):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5)
    )  # Create a single row with three subplots

    # Convert input grayscale image to RGB
    input_gray = inputimg.permute((1, 2, 0)).detach().cpu().numpy()
    # Plot input image
    axes[0].imshow(input_gray, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot target image
    target_img = targetimg.permute((1, 2, 0)).detach().cpu().numpy()
    # target_img = denormalize_lab_image(target_img)
    # target_rgb = cv2.cvtColor(target_img, cv2.COLOR_LAB2RGB)
    target_rgb = color.lab2rgb(target_img)
    axes[1].imshow(target_rgb)
    axes[1].set_title("Target Image")
    axes[1].axis("off")

    # Plot predicted image
    predict_img = predictedimg.permute((1, 2, 0)).detach().cpu().numpy()
    # predict_img = denormalize_lab_image(predict_img)
    # predicted_rgb = predict_img  # cv2.cvtColor(predict_img, cv2.COLOR_LAB2RGB)
    predicted_rgb = color.lab2rgb(predict_img)
    axes[2].imshow(predicted_rgb)
    axes[2].set_title("Predicted Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("TestImage.jpg")
    plt.show()
