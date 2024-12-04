import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
import os
from tqdm import tqdm
from PIL import Image
import json

class EllipseEstimatorModel(nn.Module):
    def __init__(self):
        super(EllipseEstimatorModel, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Reduced to 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced to 32 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced to 64 channels
        self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by half
        
        # Global Average Pooling reduces spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 64)  # Reduced to 64 units
        self.fc2 = nn.Linear(64, 1)  # Output for binary classification
        self.fc3 = nn.Linear(64, 5)  # Output for regression parameters

    def forward(self, mImg):
        # Apply convolutional and pooling layers
        mImg = self.pool(torch.relu(self.conv1(mImg)))
        mImg = self.pool(torch.relu(self.conv2(mImg)))
        mImg = self.pool(torch.relu(self.conv3(mImg)))
        
        # Apply global average pooling
        mImg = self.global_pool(mImg)  # Shape: [batch_size, 64, 1, 1]
        mImg = mImg.view(mImg.size(0), -1)  # Flatten to [batch_size, 64]

        # Apply fully connected layers
        mImg = torch.relu(self.fc1(mImg))
        mClassOutput = torch.sigmoid(self.fc2(mImg))  # Sigmoid for binary classification
        mRegOutput = self.fc3(mImg)  # Regression for ellipse parameters
        
        return mClassOutput, mRegOutput

# Define the CNN model
class EllipseEstimatorModel_old(nn.Module):
    def __init__(self):
        super(EllipseEstimatorModel, self).__init__()
        
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by half
        
        # Calculate flattened size after conv and pooling
        self.flattened_size = 64 * (400 // 4) * (400 // 4)  # (400 is input size, //4 accounts for 2 pooling layers)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Output for binary classification
        self.fc3 = nn.Linear(128, 5)  # Output for regression parameters

    def forward(self, mImg):
        mImg = self.pool(torch.relu(self.conv1(mImg)))
        mImg = self.pool(torch.relu(self.conv2(mImg)))
        mImg = mImg.view(mImg.size(0), -1)  # Flatten

        mImg = self.dropout(torch.relu(self.fc1(mImg)))
        mClassOutput = torch.sigmoid(self.fc2(mImg))  # Sigmoid for binary classification
        mRegOutput = self.fc3(mImg)  # Regression for ellipse parameters
        return mClassOutput, mRegOutput

# Dataset class 
class EllipseDataset(Dataset):
    def __init__(self, lImagePaths, lLabels, lParamsMean, lParamsSTD, transform=None):
        """
        Args:
            lImagePaths (list): List of image file paths.
            lLabels (list): List of labels (0 or 1) for classification.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.lImagePaths = lImagePaths
        self.lLabels     = lLabels
        self.paramsMean  = lParamsMean
        self.paramsStd   = lParamsSTD
        self.transform   = transform

    def __len__(self):
        return len(self.lImagePaths)

    def __getitem__(self, idx):

        imagePath = self.lImagePaths[idx]
        # Read the corresponding JSON file
        jsonPath = self.lImagePaths[idx].rsplit('.', 1)[0] + '.json'
        
        with open(jsonPath, 'r') as jsonFile:
            dData = json.load(jsonFile)

        # Extract ellipse parameters
        label = lLabels[idx]
        tLabel = torch.tensor(label, dtype=torch.float32)

        # If an ellipse exists, extract parameters; otherwise, set all parameters to zero
        if label:
            vTarget = dData["centerCoord"] + dData["radLength"] + [dData["rotAngle"]]
        else:
            vTarget = [0.0, 0.0, 0.0, 0.0, 0.0]
        # Convert the target list to a tensor
        tTarget = torch.tensor(vTarget, dtype=torch.float32)
        tTarget = (tTarget - torch.tensor(self.paramsMean, dtype=torch.float32)) / torch.tensor(self.paramsStd, dtype=torch.float32) # Standardization *for theta we only divide by 3.14 (max)


        # Apply any transformations to the image (e.g., normalization)
         # Read the image
        mImg = Image.open(imagePath).convert("RGB")  # Assuming image is stored in the specified path
        if self.transform:
            tImg = self.transform(mImg)
            
        
        return tImg, tLabel, tTarget

# Loss functions
# Binary Cross-Entropy for classification
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, mInputs, mTargets):
        return nn.BCEWithLogitsLoss()(mInputs, mTargets)

# Training function
def train_model(mModel, mTrainLoader, mValLoader, mDevice, positiveWeight=1, learningRate=1e-4, epochs=10):
   # Phase 1: Train Classification
    print("Phase 1: Training Classification Branch")
    set_trainable_layers(mModel, train_classification=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mModel.parameters()), lr=1e-3, weight_decay=1e-4)
    criterionCls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5]).to(device))

    for epoch in range(10):  # Adjust epochs as needed
        mModel.train()
        for mInputs, vTarget in mTrainLoader:
            mInputs = mInputs.to(mDevice)
            mLabels = vTarget[:, 0].to(mDevice)

            # Forward pass
            outputsCls, _ = mModel(mInputs)
            lossCls = criterionCls(outputsCls, mLabels)

            # Backward pass
            optimizer.zero_grad()
            lossCls.backward()
            optimizer.step()

        # Optional: Add validation logic here

        # Phase 2: Train Regression
        print("Phase 2: Training Regression Branch")
        set_trainable_layers(mModel, train_classification=False)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mModel.parameters()), lr=1e-3, weight_decay=1e-4)
        criterionReg = nn.MSELoss()

    for epoch in range(epochs):  # Adjust epochs as needed
        mModel.train()
        for mInputs, vTarget in mTrainLoader:
            mInputs = mInputs.to(mDevice)
            mRegLabels = vTarget[:, 1:].to(mDevice)

            # Forward pass
            _, outputsReg = mModel(mInputs)
            lossReg = criterionReg(outputsReg, mRegLabels)

            # Backward pass
            optimizer.zero_grad()
            lossReg.backward()
            optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss (Class): {mAvgLossClass:.4f}, Train Loss (Reg): {mAvgLossReg:.4f}, Val Loss (Class): {mValLossClass:.4f}, Val Loss (Reg): {mValLossReg:.4f}, Accuracy: {mAccuracy:.4f}")

    # Plot losses
    plot_losses(lTrainLossClass, lTrainLossReg, lValLossClass, lValLossReg)
    return mModel

# Evaluation function
def evaluate_model(mModel, mDataLoader, mDevice):
    mModel.eval()
    mRunningLossClass = 0.0
    mRunningLossReg = 0.0
    mCorrect = 0
    mTotal = 0

    with torch.no_grad():
        for mInputs, mLabels, mRegLabels in mDataLoader:
            mInputs, mLabels, mRegLabels = mInputs.to(mDevice), mLabels.to(mDevice), mRegLabels.to(mDevice)

            # Forward pass
            mClassOutput, mRegOutput = mModel(mInputs)
            
            # Classification loss
            mLossClass = nn.BCEWithLogitsLoss()(mClassOutput, mLabels.unsqueeze(1))
            # Regression loss
            mLossReg = nn.MSELoss()(mRegOutput, mRegLabels)

            mRunningLossClass += mLossClass.item()
            mRunningLossReg += mLossReg.item()

            mPredictedClass = mClassOutput >= 0.5
            mCorrect += (mPredictedClass == mLabels.unsqueeze(1)).sum().item()
            mTotal += mLabels.size(0)

    mAvgLossClass = mRunningLossClass / len(mDataLoader)
    mAvgLossReg = mRunningLossReg / len(mDataLoader)
    mAccuracy = mCorrect / mTotal
    return mAvgLossClass, mAvgLossReg, mAccuracy

# Plotting function for losses
def plot_losses(mTrainLossClass, mTrainLossReg, mValLossClass, mValLossReg):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mTrainLossClass, label='Train Loss (Class)')
    plt.plot(mValLossClass, label='Val Loss (Class)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mTrainLossReg, label='Train Loss (Reg)')
    plt.plot(mValLossReg, label='Val Loss (Reg)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def ListDataPaths(folderPath):
    lImagePaths = []
    lJsonPaths = []
    lLabels = []

    for fileName in os.listdir(folderPath):
        if fileName.endswith(('.jpg', '.jpeg', '.png')):
            imagePath = os.path.join(folderPath, fileName)
            jsonFile = fileName.rsplit('.', 1)[0] + '.json'
            jsonPath = os.path.join(folderPath, jsonFile)

            if os.path.exists(jsonPath):
                lImagePaths.append(imagePath)
                lJsonPaths.append(jsonPath)

                with open(jsonPath, 'r') as json_file:
                    mData = json.load(json_file)
                    mIsEllipse = mData.get("isEllipse", False)
                    lLabels.append(1 if mIsEllipse else 0)
    return lImagePaths, lLabels

def CalculateDatasetStatistics(imageFolderPath):
    """
    Calculates the mean and standard deviation of image channels (R, G, B) 
    and ellipse parameters (x, y, a, b) from the dataset.
    
    Parameters:
        imageFolderPath (str): Path to the folder containing images and JSON files.

    Returns:
        dict: Dictionary with calculated statistics for images and ellipse parameters.
    """
    # Initialize accumulators for image stats
    sumR, sumG, sumB = 0.0, 0.0, 0.0
    sumSqR, sumSqG, sumSqB = 0.0, 0.0, 0.0
    nPixels = 0

    # Initialize accumulators for ellipse parameter stats
    lX, lY, lA, lB, lLabels = [], [], [], [], []

    # Gather image and JSON file paths
    lImgPaths  = [os.path.join(imageFolderPath, file) for file in os.listdir(imageFolderPath) if file.endswith(('.jpg', '.png'))]
    lJsonPaths = [path.replace('.png', '.json').replace('.jpg','.json') for path in lImgPaths]

    for imgPath, jsonPath in tqdm(zip(lImgPaths, lJsonPaths), desc="Processing files", total=len(lImgPaths)):
        # Process the image
        mImg = Image.open(imgPath).convert('RGB')
        mImg = np.array(mImg) / 255.0  # Normalize to range [0, 1]

        # Accumulate channel sums and squared sums
        sumR += np.sum(mImg[:, :, 0])
        sumG += np.sum(mImg[:, :, 1])
        sumB += np.sum(mImg[:, :, 2])

        sumSqR += np.sum(mImg[:, :, 0] ** 2)
        sumSqG += np.sum(mImg[:, :, 1] ** 2)
        sumSqB += np.sum(mImg[:, :, 2] ** 2)

        # Update total pixel count
        nPixels += mImg.shape[0] * mImg.shape[1]

        # Process the JSON file
        if os.path.exists(jsonPath):
            with open(jsonPath, 'r') as f:
                data = json.load(f)
                lLabels.append(1 if data["isEllipse"] else 0)
                if data["isEllipse"]:
                    lX.append(data["centerCoord"][0])
                    lY.append(data["centerCoord"][1])
                    lA.append(data["radLength"][0])
                    lB.append(data["radLength"][1])

    # Calculate mean and std for images
    imageMean = [sumR / nPixels, sumG / nPixels, sumB / nPixels]
    imageStd = [
        np.sqrt(sumSqR / nPixels - imageMean[0] ** 2),
        np.sqrt(sumSqG / nPixels - imageMean[1] ** 2),
        np.sqrt(sumSqB / nPixels - imageMean[2] ** 2),
    ]

    # Calculate mean and std for ellipse parameters
    meanParams = [np.mean(lX), np.mean(lY), np.mean(lA), np.mean(lB), 0]  # Theta mean is 0
    stdParams = [np.std(lX), np.std(lY), np.std(lA), np.std(lB), 3.14]    # Theta std is 3.14

    dDataStatistics = {
        "imageMean": imageMean,
        "imageStd": imageStd,
        "paramsMean": meanParams,
        "paramsStd": stdParams
    }

    return lImgPaths, lLabels, dDataStatistics 

def reset_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()

def set_trainable_layers(model, train_classification=True):
    """
    Freezes and unfreezes layers based on the training phase.
    Args:
        model: The model to modify.
        train_classification: If True, train classification layers only.
                              If False, train regression layers only.
    """
    for param in model.parameters():
        param.requires_grad = True  # Enable all gradients initially

    if train_classification:
        for param in model.fc3.parameters():  # Freeze regression branch
            param.requires_grad = False
    else:
        for param in model.fc2.parameters():  # Freeze classification branch
            param.requires_grad = False


# Main function
if __name__ == '__main__':
    # Load your dataset paths, labels, and regression data here
    dataFolderPath = 'Data'  # path to images folder

    #Hyperparameters
    batchSize = 16
    learningRate = 1e-4
    numOfEpochs = 25
    valSize = 0.2
    rndSeed = 42

    
    #lImagePaths, lLabels = ListDataPaths(dataFolderPath) Not in use
    lImagePaths, lLabels, dDataStatistics = CalculateDatasetStatistics(dataFolderPath)

    positiveWeight = 1 #(len(lLabels) - np.sum(lLabels))/np.sum(lLabels)
    # 2. Perform stratified train-validation split
    lTrainImagePaths, lValImagePaths, lTrainLabels, lValLabels = train_test_split(lImagePaths, lLabels, test_size=valSize, stratify=lLabels, random_state=rndSeed)

    # 3. Create DataLoader instances for training and validation datasets
    transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=dDataStatistics["imageMean"], std=dDataStatistics["imageStd"]),  # Image normalization
                ])

    paramsMean = dDataStatistics['paramsMean']
    paramStd   = dDataStatistics['paramsStd']

    tTrainDataset = EllipseDataset(list(lTrainImagePaths), list(lTrainLabels), paramsMean, paramStd, transform=transform)
    tValDataset   = EllipseDataset(list(lValImagePaths), list(lValLabels), paramsMean, paramStd, transform=transform)

   
        # Set device (GPU or CPU)
    mDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_counts = [len(lLabels)-np.sum(lLabels), np.sum(lLabels)]  # [Minority count (isEllipse = 0), Majority count (isEllipse = 1)]

    # Compute class weights (inverse frequency)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # Assign a weight to each sample based on its class
    sample_weights = [class_weights[int(label)] for _, label, _ in tTrainDataset]

    # Create a sampler with replacement to oversample the minority class
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    mTrainLoader = DataLoader(tTrainDataset, batch_size=batchSize, shuffle=True)
    mValLoader = DataLoader(tValDataset, batch_size=batchSize, shuffle=False)

    # Initialize model
    mModel = EllipseEstimatorModel().to(mDevice)
    mModel.apply(reset_weights)
    positiveWeight = torch.tensor(0.5).to(mDevice) # positiveWeight

     # Train the model
    mModel = train_model(mModel, mTrainLoader, mValLoader, mDevice, positiveWeight, epochs=numOfEpochs)

