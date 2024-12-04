import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


# Dataset Definition
class EllipseDataset(Dataset):
    def __init__(self, image_dir, json_dir, image_list, transform=None):
        """
        Args:
            image_dir (str): Directory containing all images.
            json_dir (str): Directory containing all JSON files.
            image_list (list): List of image filenames (excluding paths).
            transform (callable, optional): Transform to apply to the images.
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        json_path = os.path.join(self.json_dir, image_name.replace(".png", ".json"))

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load label from JSON
        with open(json_path, "r") as f:
            metadata = json.load(f)
        label = 1.0 if metadata["isEllipse"] else 0.0

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

# Model Definition
class EllipseClassifier(nn.Module):
    def __init__(self):
        super(EllipseClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.global_pool(x).view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Training Function

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    # Lists to store training and validation losses
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate validation loss and accuracy
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # Append losses and accuracy for plotting
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optionally return tracked losses and accuracies
    return train_losses, val_losses, val_accuracies

# Evaluation Function
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return val_loss / len(val_loader), correct / total

# Main Script
if __name__ == "__main__":
    # Dataset paths
    data_dir = "ellipse_data"  # Path to the folder with images
    json_dir = "ellipse_data"  # Path to the folder with JSON files (same as data_dir in this case)

    # List of all image files
    image_list = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    labels = []
    for img in image_list:
        json_path = os.path.join(json_dir, img.replace(".png", ".json"))
        with open(json_path, "r") as f:
            metadata = json.load(f)
            labels.append(1 if metadata["isEllipse"] else 0)

    train_list, val_list, train_labels, val_labels = train_test_split(
    image_list, labels, test_size=0.2, random_state=42, stratify=labels
    )   

    # Transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Datasets and loaders
    train_dataset = EllipseDataset(data_dir, json_dir, train_list, transform=transform)
    val_dataset = EllipseDataset(data_dir, json_dir, val_list, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EllipseClassifier().to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=300)

