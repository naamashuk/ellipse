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

# Training Function with Checkpointing
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoint_path=None):
    best_val_loss = float('inf')  # Initialize the best validation loss
    best_model_state = None

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
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

            if checkpoint_path:
                torch.save(best_model_state, checkpoint_path)  # Save the best model to file

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    return model  # Return the best model

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
    # Dataset paths for high-SNR data (Data 1)
    data1_dir = "ellipse_data"
    json1_dir = "ellipse_data"

    # Dataset paths for low-SNR data (Data 2)
    data2_dir = "ellipse_data"
    json2_dir = "ellipse_data"

    # List of image files for Data 1
    image_list1 = [f for f in os.listdir(data1_dir) if f.endswith(".png")]
    labels1 = []
    for img in image_list1:
        json_path = os.path.join(json1_dir, img.replace(".png", ".json"))
        with open(json_path, "r") as f:
            metadata = json.load(f)
            labels1.append(1 if metadata["isEllipse"] else 0)

    # List of image files for Data 2
    image_list2 = [f for f in os.listdir(data2_dir) if f.endswith(".png")]
    labels2 = []
    for img in image_list2:
        json_path = os.path.join(json2_dir, img.replace(".png", ".json"))
        with open(json_path, "r") as f:
            metadata = json.load(f)
            labels2.append(1 if metadata["isEllipse"] else 0)

    # Train-validation split for Data 1
    train_list1, val_list1, train_labels1, val_labels1 = train_test_split(
        image_list1, labels1, test_size=0.2, random_state=42, stratify=labels1
    )

    # Train-validation split for Data 2
    train_list2, val_list2, train_labels2, val_labels2 = train_test_split(
        image_list2, labels2, test_size=0.2, random_state=42, stratify=labels2
    )

    # Transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # DataLoaders
    train_dataset1 = EllipseDataset(data1_dir, json1_dir, train_list1, transform=transform)
    val_dataset1 = EllipseDataset(data1_dir, json1_dir, val_list1, transform=transform)
    train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
    val_loader1 = DataLoader(val_dataset1, batch_size=32, shuffle=False)

    train_dataset2 = EllipseDataset(data2_dir, json2_dir, train_list2, transform=transform)
    val_dataset2 = EllipseDataset(data2_dir, json2_dir, val_list2, transform=transform)
    train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True)
    val_loader2 = DataLoader(val_dataset2, batch_size=32, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EllipseClassifier().to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train on Data 1
    print("Training on Data 1 (High-SNR)...")
    #best_model_path = "best_model_high_snr.pth"
    #model = train_model(model, train_loader1, val_loader1, criterion, optimizer, device, epochs=25, checkpoint_path=best_model_path)

    # Fine-tune on Data 2
    print("Fine-tuning on Data 2 (Low-SNR)...")
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    for param in model.fc1.parameters():
        param.requires_grad = True  # Unfreeze fully connected layers
    for param in model.fc2.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    model = train_model(model, train_loader2, val_loader2, criterion, optimizer, device, epochs=40)
