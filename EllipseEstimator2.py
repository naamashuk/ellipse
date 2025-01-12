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
import torch.nn.init as init
import cv2
import random
from PIL import Image
from draw_ellipse import draw_points_on_image




def generate_and_save_ellipse_comparison(image_size, gt_params, pred_params, save_dir, run_index, thickness=2, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """
    Generates an image with ground truth and predicted ellipses and saves it as a PNG file.
    
    Parameters:
    - image_size: Tuple specifying the size of the image (height, width)
    - gt_params: Ground truth parameters [x, y, a, b, theta]
    - pred_params: Predicted parameters [x, y, a, b, theta]
    - save_dir: Directory to save the PNG image
    - run_index: Index to name the file
    - thickness: Thickness of the ellipse contours (default: 2)
    - gt_color: Color of the ground truth ellipse (default: green)
    - pred_color: Color of the predicted ellipse (default: red)
    
    Returns:
    - file_path: Path to the saved PNG image
    """
    

    # Create a blank image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # Ground truth ellipse
    gt_center = (int(gt_params[0]), int(gt_params[1]))
    gt_axes = (int(gt_params[2]), int(gt_params[3]))
    gt_angle = np.degrees(gt_params[4])  # Convert radians to degrees
    cv2.ellipse(image, gt_center, gt_axes, gt_angle, 0, 360, gt_color, thickness)
    
    # Predicted ellipse
    pred_center = (int(pred_params[0]), int(pred_params[1]))
    pred_axes = (int(pred_params[2]), int(pred_params[3]))
    pred_angle = np.degrees(pred_params[4])  # Convert radians to degrees
    cv2.ellipse(image, pred_center, pred_axes, pred_angle, 0, 360, pred_color, thickness)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the image
    filename = f"ellipse_comparison_{run_index}.png"
    file_path = os.path.join(save_dir, filename)
    cv2.imwrite(file_path, image)
    
    #print(f"Image saved to: {file_path}")
    return file_path



def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight)
        #init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)

def normalize_params(params, x_max, y_max, a_max, b_max, theta_max):
    """
    Normalize ellipse parameters to the range [0, 1].
    
    Args:
        params (numpy array): Parameters [x, y, a, b, theta]. Can be 1D (for a single sample) or 2D.
        x_max, y_max: Maximum values for x and y.
        a_max, b_max: Maximum values for a and b (ellipse radii).
        theta_max: Maximum value for theta (typically pi).
    
    Returns:
        Normalized parameters.
    """
    params = np.atleast_2d(params)  # Ensure params is 2D
    x, y, a, b, theta = params.T
    x_norm = x / x_max
    y_norm = y / y_max
    a_norm = a / a_max
    b_norm = b / b_max
    theta_norm = theta / theta_max
    return np.stack([x_norm, y_norm, a_norm, b_norm, theta_norm], axis=1)


def denormalize_params(params_norm, x_max, y_max, a_max, b_max, theta_max):
    """
    Denormalize parameters from the range [0, 1] back to their original scale.

    Args:
        params_norm (numpy array): Normalized parameters [x, y, a, b, theta].
        x_max, y_max: Maximum values for x and y.
        a_max, b_max: Maximum values for a and b (ellipse radii).
        theta_max: Maximum value for theta (typically pi).

    Returns:
        Denormalized parameters.
    """
    x_norm, y_norm, a_norm, b_norm, theta_norm = params_norm.T
    x = x_norm * x_max
    y = y_norm * y_max
    a = a_norm * a_max
    b = b_norm * b_max
    theta = theta_norm * theta_max
    return np.stack([x, y, a, b, theta], axis=1)


# Dataset Definition
class EllipseDataset(Dataset):
    def __init__(self, image_dir, json_dir, image_list, transform=None, param_max=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_list = image_list
        self.transform = transform
        self.param_max = param_max  # Dictionary with max values for normalization

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        #print(image_name,'--------------------------------------------------------------')
        image_path = os.path.join(self.image_dir, image_name)
        json_path = os.path.join(self.json_dir, image_name.replace(".png", ".json"))

        # Load image
        image = Image.open(image_path)

        # Load parameters
        with open(json_path, "r") as f:
            metadata = json.load(f)

        label = 1.0 if metadata["isEllipse"] else 0.0
        
        center = metadata["centerCoord"]
        radii = metadata["radLength"]
        theta = metadata["rotAngle"]

        # Combine and normalize parameters
        reg_params = np.array([center[0], center[1], radii[0], radii[1], theta])
        if label > 0.5:
            
            black_image = np.zeros((400,400,3), dtype=np.uint8)
            point_color = (255,255,0)  # White color for binary image (grayscale)
            point_radius = 2  # Radius of each point
            thickness = -1  # Fill the points (circle)
            
            if(0):
                cv2.ellipse(black_image, (center[0], center[1]), (radii[0], radii[1]), int(theta/(np.pi)*180), 0, 360, (255,0,0), 2)

                # Save the image
                output_path = image_path.replace('.', '_debug.')  # Replace with your desired path
                cv2.imwrite(output_path, black_image)

            reg_params = normalize_params(
                reg_params,
                x_max=self.param_max["x"],
                y_max=self.param_max["y"],
                a_max=self.param_max["a"],
                b_max=self.param_max["b"],
                theta_max=self.param_max["theta"]
            ).flatten()  # Ensure 1D shape after normalization

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), torch.tensor(reg_params, dtype=torch.float32),image_name


def combined_loss(class_output, reg_output, labels, reg_labels, sampled_points, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
    """
    Combined loss for classification and regression with geometric and shape penalties.
    """
    # Classification loss (weighted BCE)
    loss_cls = -alpha * labels * torch.log(class_output + 1e-6) - (1 - labels) * torch.log(1 - class_output + 1e-6)
    loss_cls = loss_cls.mean()

    # Regression loss (only for positive samples)
    mask = labels > 0.5
    loss_reg = torch.tensor(0.0)
    penalty_geo = torch.tensor(0.0)
    penalty_shape = torch.tensor(0.0)

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
        penalty_shape = torch.abs(ellipse_eq - 1).mean() / sampled_points.size(1)  # Normalize by the number of points

    # Total loss
    total_loss = alpha * loss_cls + beta * loss_reg + gamma * penalty_geo + delta * penalty_shape

    return total_loss, loss_cls, loss_reg, penalty_geo, penalty_shape



class EllipseDetector(nn.Module):
    def __init__(self):
        super(EllipseDetector, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

        self.fc3 = nn.Linear(625,256)

        self.fc4 = nn.Linear(256,5)
        

    def forward(self, x):
        x1 = self.pool(torch.relu(self.conv1(x)))
        x1 = self.pool(torch.relu(self.conv2(x1)))

        x_features = self.pool(torch.relu(self.conv3(x1)))
        x_cl = self.global_pool(x_features).view(x_features.size(0), -1)
        
        x_cl = torch.relu(self.fc1(x_cl))
        x_cl = torch.sigmoid(self.fc2(x_cl))

        # is ellipse
        x_reg = torch.relu(self.pool(self.conv4(x_features)))
        x_reg = torch.relu(self.fc3(x_reg.view(x_reg.size(0), -1)))
        x_reg = torch.relu(self.fc4(x_reg))

        return x_cl, x_reg


def sample_points_from_gt(gt_params, num_points=100):
    """
    Sample points directly from the ground truth ellipse.

    Args:
        gt_params (Tensor): Ground truth parameters [x0, y0, a, b, theta] (batch_size, 5).
        num_points (int): Number of points to sample for each ellipse.

    Returns:
        sampled_points (Tensor): Sampled points on the GT ellipse (batch_size, num_points, 2).
    """
    batch_size = gt_params.size(0)
    angles = torch.linspace(0, 2 * torch.pi, num_points, device=gt_params.device)  # (num_points)
    angles = angles.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, num_points)

    # Extract GT parameters
    x0, y0, a, b, theta = gt_params.T  # (batch_size)

    # Compute ellipse points
    x = x0.unsqueeze(1) + a.unsqueeze(1) * torch.cos(angles) * torch.cos(theta*2*np.pi).unsqueeze(1) - \
        b.unsqueeze(1) * torch.sin(angles) * torch.sin(theta*2*np.pi).unsqueeze(1)
    y = y0.unsqueeze(1) + a.unsqueeze(1) * torch.cos(angles) * torch.sin(theta*2*np.pi).unsqueeze(1) + \
        b.unsqueeze(1) * torch.sin(angles) * torch.cos(theta*2*np.pi).unsqueeze(1)
    
    #plot ellipse
    if(0):
        if x.sum() >0 and y.sum() > 0:
            black_image = np.zeros((400,400,3), dtype=np.uint8)
            point_color = (255,255,0)  # White color for binary image (grayscale)
            point_radius = 2  # Radius of each point
            thickness = -1  # Fill the points (circle)
            for i in range(x.shape[1]):
                cv2.circle(black_image, (int(x[0,i]*400), int(y[0,i]*400)), point_radius, point_color, thickness)
            
            cv2.ellipse(black_image, (int(x0*400),int(y0*400)), (int(a*400),int(b*400)), int(theta*2*np.pi/(np.pi)*180), 0, 360, (255,0,0), 2)

            # Save the image
            output_path = 'debug/ellipse_from_points.png'  # Replace with your desired path
            cv2.imwrite(output_path, black_image)

    # Combine x and y into sampled points
    sampled_points = torch.stack((x, y), dim=-1)  # (batch_size, num_points, 2)
    return sampled_points



def train_model(
    model, train_loader, val_loader, optimizer, device, param_means, param_stds,
    epochs=150, checkpoint_dir=None, log_file=None, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
    
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

        for images, labels, reg_labels, image_name in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels, reg_labels = images.to(device), labels.to(device), reg_labels.to(device)

            # Generate sampled points for the batch
            sampled_points = sample_points_from_gt(reg_labels, num_points=100)


            # fig, ax = plt.subplots(1,1, num=3)
            # ax.plot(sampled_points.numpy().squeeze()[:,0]*400, -400*sampled_points.numpy().squeeze()[:,1], '.')
            # ax.set_xlim(0, 400)
            # ax.set_ylim(-400, 0)
            # fig.savefig('./debug' + '/' + image_name[0])

            # Forward pass
            if labels[0] == 0:
                continue
            class_output, reg_output = model(images)

            # Combined loss
            loss, loss_cls, loss_reg, penalty_geo, penalty_shape = combined_loss(
                class_output, reg_output, labels, reg_labels, sampled_points, alpha, beta, gamma, delta*int(epoch>8)
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss_cls += loss_cls.item()
            train_loss_reg += loss_reg.item()

        # Validation
        # Validation
        val_loss_cls, val_loss_reg, val_acc, ratios = evaluate_model(
            model, val_loader, nn.BCELoss(), nn.MSELoss(), device, param_means, param_stds
        )

        # Calculate mean error across all parameters
        mean_error = sum(ratios.values()) / len(ratios)

        # Print metrics for the current epoch
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_total / len(train_loader):.4f}, "
            f"Train Loss (Cls): {train_loss_cls / len(train_loader):.4f}, "
            f"Train Loss (Reg): {train_loss_reg / len(train_loader):.4f}, "
            f"Val Loss (Cls): {val_loss_cls:.4f}, Val Loss (Reg): {val_loss_reg:.4f}, "
            f"Val Accuracy: {val_acc:.4f}, Mean Error: {mean_error:.4f}")
        print(f"Ratios (Val): x: {ratios['x']:.4f}, y: {ratios['y']:.4f}, "
            f"a: {ratios['a']:.4f}, b: {ratios['b']:.4f}, theta: {ratios['theta']:.4f}")

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

        # Export metrics to CSV
        if log_file:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss_total / len(train_loader),
                    train_loss_cls / len(train_loader),
                    train_loss_reg / len(train_loader),
                    val_loss_cls,
                    val_loss_reg,
                    val_acc,
                    mean_error,
                    ratios['x'],
                    ratios['y'],
                    ratios['a'],
                    ratios['b'],
                    ratios['theta']
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
        for images, labels, reg_labels, image_name in val_loader:
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

                # Reverse normalization
                pred_params = reg_output[mask].cpu().numpy()
                gt_params = reg_labels[mask].cpu().numpy()
                pred_params_orig = pred_params * [400,400,400,400,2*np.pi]
                gt_params_orig = gt_params * [400,400,400,400,2*np.pi]

                # Example usage
                # image_size = (400, 400)
                # gt_params = [200, 200, 100, 50, np.pi / 4]  # Ground truth parameters
                # pred_params = [200, 200, 90, 60, np.pi / 3]  # Predicted parameters
                save_dir = "./ellipse_images"
                # run_index = 1

                param_max = {
                "x": 400,  # Image width
                "y": 400,  # Image height
                "a": 400,  # Max semi-major axis
                "b": 400,  # Max semi-minor axis
                "theta": 2 * np.pi  # Theta max value
                }

                list_length = 10
                random_index = random.randint(0, list_length - 1)
                # Call the function
                for ii in range(gt_params_orig.shape[0]):
                    generate_and_save_ellipse_comparison((400,400), gt_params_orig[ii], pred_params_orig[ii], save_dir, ii)

                # Compute ratios
                for i, key in enumerate(["x", "y", "a", "b", "theta"]):
                    ratios[key] += (abs(pred_params_orig[:, i] - gt_params_orig[:, i]) / 
                                    (gt_params_orig[:, i] + 1e-6)).mean()
                count_ratios += mask.sum().item()

            # Classification accuracy
            predictions = (class_output > 0.5).float()
            correct += (predictions.squeeze(1) == labels).sum().item()
            total += labels.size(0)

    # Average ratios
    for key in ratios:
        ratios[key] /= max(1, count_ratios)  # Avoid division by zero

    return val_loss_cls / len(val_loader), val_loss_reg / max(1, count_ratios), correct / total, ratios

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



def calculate_dataset_mean_std(image_dir):
    """
    Calculate the mean and standard deviation of a dataset of images.
    Assumes images are in the specified directory, and converts pixel values to the range [0, 1].
    
    Parameters:
    - image_dir: Directory containing the training set images.
    
    Returns:
    - mean: Mean of pixel values across all images.
    - std: Standard deviation of pixel values across all images.
    """
    pixel_values = []
    lmean = []
    lstd = []
    # Iterate over all images in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Load the image
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)#.convert('RGB')  # Ensure RGB format
            image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            
            # Flatten and append pixel values
            lmean.append(image_array.reshape(-1, 3).mean())  # Flatten per channel
            lstd.append(image_array.reshape(-1, 3).std())

    # Stack all pixel values and calculate mean and std
    # all_pixels = np.vstack(pixel_values)  # Combine all pixel data
    mean = np.mean(lmean, axis=0)  # Mean per channel
    std = np.mean(lstd, axis=0)    # Std per channel
    
    return mean, std





 # Main Script
if __name__ == "__main__":
    # Dataset paths
    data_dir = "ellipse_data2"
    json_dir = "ellipse_data2"

    #mean, std = calculate_dataset_mean_std(data_dir)
    mean = [0.497, 0.497]
    std = [0.11, 0.11] # wrong remove

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
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean[0], mean[0], mean[0]], std=[std[0], std[0], std[0]])  # Example image normalization
    ])

    # Define max values based on your data
    param_max = {
        "x": 400,  # Image width
        "y": 400,  # Image height
        "a": 400,  # Max semi-major axis
        "b": 400,  # Max semi-minor axis
        "theta": 2 * np.pi  # Theta max value
    }

    # Create datasets
    train_dataset = EllipseDataset(data_dir, json_dir, train_list, transform=transform, param_max=param_max)
    val_dataset = EllipseDataset(data_dir, json_dir, val_list, transform=transform, param_max=param_max)

    # Count positive and negative samples
    # positive_count = 0
    # negative_count = 0

    # for _, label, _ in train_dataset:
    #     if label == 1.0:
    #         positive_count += 1
    #     else:
    #         negative_count += 1

    # print(f"Number of positive samples (ellipse): {positive_count}")
    # print(f"Number of negative samples (no ellipse): {negative_count}")

    for idx in range(5):  # Test first 5 samples
        image, label, reg_params, image_name = train_dataset[idx]
        print(f"Sample {idx}: reg_params shape = {reg_params.shape}, label = {label}")


    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for images, labels, reg_labels, image_name in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Regression labels shape: {reg_labels.shape}")
        break

    # Model, loss functions, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EllipseDetector()
    model.apply(initialize_weights)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params}")

    model.to(device)  # Updated model with regression branch
    #criterion_cls = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
    #criterion_reg = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Checkpoints and logging
    checkpoint_dir = "checkpoints_24_12_24"
    log_file = "training_log_24_12_24.csv"

    # Train the model
    train_model(
        model, train_loader, val_loader, optimizer, device, param_means, param_stds,
        epochs=450, checkpoint_dir=checkpoint_dir, log_file=log_file, alpha=1.0, beta=2, gamma=1, delta=3
    )

   
    # Plot training metrics
    plot_training_log(log_file)
