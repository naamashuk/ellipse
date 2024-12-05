import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# Dataset Definition
class EllipseDataset(Dataset):
    def __init__(self, image_dir, json_dir, image_list, transform=None, param_means=None, param_stds=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_list = image_list
        self.transform = transform
        self.param_means = param_means
        self.param_stds = param_stds

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        json_path = os.path.join(self.json_dir, image_name.replace(".png", ".json"))

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load label and regression parameters
        with open(json_path, "r") as f:
            metadata = json.load(f)

        label = 1.0 if metadata["isEllipse"] else 0.0
        center = metadata["centerCoord"]
        radii = metadata["radLength"]
        theta = metadata["rotAngle"]

        # Combine parameters
        reg_params = [center[0], center[1], radii[0], radii[1], theta]

        # Apply standardization
        if label > 0.5 and self.param_means is not None and self.param_stds is not None:
            reg_params = (np.array(reg_params) - self.param_means) / self.param_stds

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32), torch.tensor(reg_params, dtype=torch.float32)
    

def combined_loss(class_output, reg_output, labels, reg_labels, sampled_points, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
    """
    Combined loss for classification and regression with geometric and shape penalties.
    """
    # Classification loss (weighted BCE)
    loss_cls = -alpha * labels * torch.log(class_output + 1e-6) - (1 - labels) * torch.log(1 - class_output + 1e-6)
    loss_cls = loss_cls.mean()

    # Regression loss (only for positive samples)
    mask = labels > 0.5
    loss_reg = 0.0
    penalty_geo = 0.0
    penalty_shape = 0.0

    if mask.sum() > 0:
        # Subset for positive samples
        pred_params = reg_output[mask]  # Shape: [num_positive, 5]
        x, y, a, b, theta = pred_params.T  # Shape: [num_positive]

        # Subset regression labels
        true_params = reg_labels[mask]

        # Subset sampled_points for positive samples
        sampled_points_pos = sampled_points[mask]  # Shape: [num_positive, 100, 2]
        sampled_x, sampled_y = sampled_points_pos[..., 0], sampled_points_pos[..., 1]  # Shape: [num_positive, 100]

        # Regression loss
        loss_reg = nn.MSELoss()(pred_params, true_params)

        # Geometric constraints penalty
        penalty_geo = torch.relu(b - a).mean() + torch.relu(-a).mean() + torch.relu(-b).mean()

        # Shape regularization
        x_rot = (sampled_x - x.unsqueeze(-1)) * torch.cos(theta.unsqueeze(-1)) + (sampled_y - y.unsqueeze(-1)) * torch.sin(theta.unsqueeze(-1))
        y_rot = (sampled_y - y.unsqueeze(-1)) * torch.cos(theta.unsqueeze(-1)) - (sampled_x - x.unsqueeze(-1)) * torch.sin(theta.unsqueeze(-1))
        ellipse_eq = (x_rot / (a.unsqueeze(-1) + 1e-6))**2 + (y_rot / (b.unsqueeze(-1) + 1e-6))**2
        penalty_shape = torch.abs(ellipse_eq - 1).mean()

    # Total loss
    total_loss = alpha * loss_cls + beta * loss_reg + gamma * penalty_geo + delta * penalty_shape
    return total_loss, loss_cls, loss_reg, penalty_geo, penalty_shape



class EllipseDetector(nn.Module):
    def __init__(self):
        super(EllipseDetector, self).__init__()
        
        # Feature extractor with reduced parameters
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Input: (3, 400, 400)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, 200, 200)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 100, 100)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 50, 50)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 25, 25)
        )
        
        # Global average pooling to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output: (128, 1, 1)

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output: binary classification (ellipse or no ellipse)
            nn.Sigmoid()  # Output between 0 and 1
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Outputs [x, y, a, b, theta]
        )

    def forward(self, x):
        # Shared feature extraction
        features = self.backbone(x)
        features = self.global_pool(features)  # Shape: (batch_size, 128, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, 128)

        # Classification output
        class_out = self.class_head(features)

        # Regression output
        reg_out = self.regression_head(features)
        return class_out, reg_out


def sample_ellipse_points(batch_size, num_points=100, device="cpu"):
    """
    Generate sampled points on a unit circle and return them for regularization.

    Args:
        batch_size: Number of ellipses to sample points for.
        num_points: Number of points to sample per ellipse.
        device: Device for computation (CPU or GPU).

    Returns:
        sampled_points: Tensor of shape (batch_size, num_points, 2).
    """
    # Sample angles uniformly between 0 and 2*pi
    angles = torch.linspace(0, 2 * torch.pi, num_points, device=device).unsqueeze(0).repeat(batch_size, 1)

    # Points on the unit circle
    x = torch.cos(angles)
    y = torch.sin(angles)

    # Combine as (x, y)
    sampled_points = torch.stack((x, y), dim=-1)  # Shape: (batch_size, num_points, 2)
    return sampled_points


def train_model(
    model, train_loader, val_loader, optimizer, device, param_means, param_stds,
    epochs=50, checkpoint_dir=None, log_file=None, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 0

    # Resume training if a checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming training from epoch {start_epoch + 1}...")

    # Initialize log file
    if log_file:
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_total = 0.0
        train_loss_cls = 0.0
        train_loss_reg = 0.0

        for images, labels, reg_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels, reg_labels = images.to(device), labels.to(device), reg_labels.to(device)

            # Generate sampled points for the batch
            sampled_points = sample_ellipse_points(batch_size=len(images), num_points=100, device=device)

            # Forward pass
            class_output, reg_output = model(images)

            # Combined loss
            loss, loss_cls, loss_reg, penalty_geo, penalty_shape = combined_loss(
                class_output, reg_output, labels, reg_labels, sampled_points, alpha, beta, gamma, delta
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss_cls += loss_cls.item()
            train_loss_reg += loss_reg.item()

        # Validation
        val_loss_cls, val_loss_reg, val_acc, ratios = evaluate_model(
            model, val_loader, nn.BCELoss(), nn.MSELoss(), device, param_means, param_stds
        )

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_total / len(train_loader):.4f}, "
              f"Train Loss (Cls): {train_loss_cls / len(train_loader):.4f}, "
              f"Train Loss (Reg): {train_loss_reg / len(train_loader):.4f}, "
              f"Val Loss (Cls): {val_loss_cls:.4f}, Val Loss (Reg): {val_loss_reg:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Ratios (Val): x: {ratios['x']:.2f}, y: {ratios['y']:.2f}, a: {ratios['a']:.2f}, b: {ratios['b']:.2f}, theta: {ratios['theta']:.2f}")

        # Save the last checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, checkpoint_path)

        # Save the best model
        val_loss_total = val_loss_cls + val_loss_reg
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with total val loss: {val_loss_total:.4f}")

        # Log training metrics
        if log_file:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss_total / len(train_loader),
                    val_loss_cls,
                    val_loss_reg,
                    val_acc
                ])


# Evaluation Function
def evaluate_model(model, val_loader, criterion_cls, criterion_reg, device, param_means, param_stds):
    model.eval()
    val_loss_cls = 0.0
    val_loss_reg = 0.0
    correct = 0
    total = 0
    ratios = {"x": 0.0, "y": 0.0, "a": 0.0, "b": 0.0, "theta": 0.0}
    count_ratios = 0


    with torch.no_grad():
        for images, labels, reg_labels in val_loader:
            images, labels, reg_labels = images.to(device), labels.to(device), reg_labels.to(device)

            # Forward pass
            class_output, reg_output = model(images)

            # Classification loss
            loss_cls = criterion_cls(class_output.squeeze(), labels)
            val_loss_cls += loss_cls.item()

            # Regression loss (only for images with ellipses)
            mask = labels > 0.5
            if mask.sum() > 0:
                reg_loss = criterion_reg(reg_output[mask], reg_labels[mask])
                val_loss_reg += reg_loss.item()

                # Reverse standardization
                pred_params = reg_output[mask].cpu().numpy()
                gt_params = reg_labels[mask].cpu().numpy()
                pred_params_orig = pred_params * param_stds + param_means
                gt_params_orig = gt_params * param_stds + param_means

                # Compute ratios
                for i, key in enumerate(["x", "y", "a", "b", "theta"]):
                    ratios[key] += (abs(pred_params_orig[:, i] - gt_params_orig[:, i])/ (gt_params_orig[:, i] + 1e-6)).mean()
                count_ratios += mask.sum().item()

            # Classification accuracy
            predictions = (class_output > 0.5).float()
            predictions = predictions.squeeze(1)  # Shape becomes [32]
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Average ratios
    for key in ratios:
       ratios[key] /= count_ratios

    return val_loss_cls / len(val_loader), val_loss_reg / count_ratios if count_ratios > 0 else 0.0, correct / total, ratios




# Plot Function
def plot_training_log(log_file):
    """
    Plots the training and validation classification losses, regression losses, 
    and validation accuracy from the log file in a single figure with three horizontal subplots.

    Args:
        log_file (str): Path to the CSV file containing the training log.
    """
    # Load log file into a DataFrame
    log_data = pd.read_csv(log_file)

    # Extract columns
    epochs = log_data['Epoch']
    train_loss_cls = log_data['Train Loss (Cls)']
    train_loss_reg = log_data['Train Loss (Reg)']
    val_loss_cls = log_data['Val Loss (Cls)']
    val_loss_reg = log_data['Val Loss (Reg)']
    val_accuracy = log_data['Val Accuracy']

    # Create a single figure with 3 horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)

    # Define integer ticks
    epoch_ticks = range(1, len(epochs) + 1)

    # Plot classification losses
    axes[0].plot(epochs, train_loss_cls, label='Train Loss (Cls)', marker='o')
    axes[0].plot(epochs, val_loss_cls, label='Val Loss (Cls)', marker='x')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Classification Loss')
    axes[0].legend()
    axes[0].grid(True, axis='x')
    axes[0].set_xticks(epoch_ticks)

    # Plot regression losses
    axes[1].plot(epochs, train_loss_reg, label='Train Loss (Reg)', marker='o')
    axes[1].plot(epochs, val_loss_reg, label='Val Loss (Reg)', marker='x')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Regression Loss')
    axes[1].legend()
    axes[1].grid(True, axis='x')
    axes[1].set_xticks(epoch_ticks)

    # Plot validation accuracy
    axes[2].plot(epochs, val_accuracy, label='Val Accuracy', color='orange', marker='x')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Validation Accuracy')
    axes[2].legend()
    axes[2].grid(True, axis='x')
    axes[2].set_xticks(epoch_ticks)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def compute_standardization_stats(json_dir, image_list):
    params = []

    for image_name in image_list:
        json_path = os.path.join(json_dir, image_name.replace(".png", ".json"))
        with open(json_path, "r") as f:
            metadata = json.load(f)

        if metadata["isEllipse"]:
            center = metadata["centerCoord"]
            radii = metadata["radLength"]
            theta = metadata["rotAngle"]

            # Combine parameters
            param = [center[0], center[1], radii[0], radii[1], theta]
            params.append(param)

    params = np.array(params)
    means = params.mean(axis=0)  # Mean for each parameter
    stds = params.std(axis=0)    # Standard deviation for each parameter

    return means, stds


def reverse_standardization(pred_params, param_means, param_stds):
    return (pred_params * param_stds) + param_means


 # Main Script
if __name__ == "__main__":
    # Dataset paths
    data_dir = "ellipse_data"
    json_dir = "ellipse_data"

    # List of all image files
    image_list = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    # Read labels from JSON files
    labels = []
    for img in image_list:
        json_path = os.path.join(json_dir, img.replace(".png", ".json"))
        with open(json_path, "r") as f:
            metadata = json.load(f)
            labels.append(1 if metadata["isEllipse"] else 0)

    # Split into training and validation sets
    train_list, val_list, train_labels, val_labels = train_test_split(
        image_list, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Compute mean and standard deviation for regression parameters
    param_means, param_stds = compute_standardization_stats(json_dir, train_list)

    # Adjust theta normalization
    param_means[-1] = 0.0  # Theta mean
    param_stds[-1] = 3.14  # Theta max value (pi)

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example image normalization
    ])

    # Create datasets
    train_dataset = EllipseDataset(data_dir, json_dir, train_list, transform=transform, param_means=param_means, param_stds=param_stds)
    val_dataset = EllipseDataset(data_dir, json_dir, val_list, transform=transform, param_means=param_means, param_stds=param_stds)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, loss functions, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EllipseDetector()
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params}")

    model.to(device)  # Updated model with regression branch
    #criterion_cls = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
    #criterion_reg = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Checkpoints and logging
    checkpoint_dir = "checkpoints2"
    log_file = "training_log2.csv"

    # Train the model
    train_model(
        model, train_loader, val_loader, optimizer, device, param_means, param_stds,
        epochs=50, checkpoint_dir=checkpoint_dir, log_file=log_file, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5
    )

   
    # Plot training metrics
    plot_training_log(log_file)
