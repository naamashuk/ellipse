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
import sys
sys.path.append('C:\\Users\\Naama\\Documents\\EllipseEstimator')
from ModelsEllipse import EllipseEstimatorModel, initialize_weights
from DataClasses import EllipseDatasetClassification
import os
from tqdm import tqdm
from PIL import Image
import json



# Loss functions
# Binary Cross-Entropy for classification
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, mInputs, mTargets):
        return nn.BCEWithLogitsLoss()(mInputs, mTargets)

# Training function
def train_model(mModel, mTrainLoader, mValLoader, mDevice, positiveWeight=1, learningRate=1e-4, epochs=10):
    criterionClass = nn.BCEWithLogitsLoss(pos_weight=positiveWeight)  # BCE for classification
    optimizer = torch.optim.Adam(mModel.parameters(), lr=learningRate, weight_decay=1e-4)

    # Track losses
    lTrainLossClass = []
    lValLossClass = []
    
    for epoch in range(epochs):
        mModel.train()
        mRunningLossClass = 0.0
        mCorrect = 0
        mTotal = 0

        for mInputs, mLabels in tqdm(mTrainLoader, desc=f"Epoch {epoch+1}/{epochs}", disable=True):
            mInputs, mLabels = mInputs.to(mDevice), mLabels.to(mDevice)
            optimizer.zero_grad()

            # Forward pass
            mClassOutput = mModel(mInputs)
            
            # Classification loss
            mLossClass = criterionClass(mClassOutput, mLabels.unsqueeze(1))
            # Regression loss
            #mLossReg = criterionRegression(mRegOutput, mRegLabels)

            # Total loss
            mLoss = mLossClass #+ mLossReg
            mLoss.backward()
            optimizer.step()

            mRunningLossClass += mLossClass.item()

            # Metrics for evaluation
            mPredictedClass = mClassOutput >= 0.5
            mCorrect += (mPredictedClass == mLabels.unsqueeze(1)).sum().item()
            mTotal += mLabels.size(0)

        # Calculate average losses and accuracy for this epoch
        mAvgLossClass = mRunningLossClass / len(mTrainLoader)
        mAccuracy = mCorrect / mTotal

        lTrainLossClass.append(mAvgLossClass)

        # Validation
        mValLossClass, mValAccuracy = evaluate_model(mModel, mValLoader, mDevice)
        mValLossReg = 0 #TODO remove
        lValLossClass.append(mValLossClass)
        #lValLossReg.append(mValLossReg)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss (Class): {mAvgLossClass:.4f}, Val Loss (Class): {mValLossClass:.4f}, Train Accuracy: {mAccuracy:.4f}, Val Accuracy: {mValAccuracy:.4f}")

    # Plot losses
    plot_losses(lTrainLossClass, mAccuracy, lValLossClass, mValAccuracy )
    return mModel

# Evaluation function
def evaluate_model(mModel, mDataLoader, mDevice):
    mModel.eval()
    mRunningLossClass = 0.0
    mRunningLossReg = 0.0
    mCorrect = 0
    mTotal = 0

    with torch.no_grad():
        for mInputs, mLabels in mDataLoader:
            mInputs, mLabels = mInputs.to(mDevice), mLabels.to(mDevice)

            # Forward pass
            mClassOutput = mModel(mInputs)
            
            # Classification loss
            mLossClass = nn.BCEWithLogitsLoss()(mClassOutput, mLabels.unsqueeze(1))

            mRunningLossClass += mLossClass.item()

            mPredictedClass = mClassOutput >= 0.5
            mCorrect += (mPredictedClass == mLabels.unsqueeze(1)).sum().item()
            mTotal += mLabels.size(0)

    mAvgLossClass = mRunningLossClass / len(mDataLoader)
    mAccuracy = mCorrect / mTotal
    return mAvgLossClass, mAccuracy

# Plotting function for losses
def plot_losses(lTrainLossClass, mAccuracy, lValLossClass, mValAccuracy):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lTrainLossClass, label='Train Loss (Class)')
    plt.plot(lValLossClass, label='Val Loss (Class)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mAccuracy, label='Train Acc')
    plt.plot(mValAccuracy, label='Val Acc')
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

# Main function
if __name__ == '__main__':
    # Load your dataset paths, labels, and regression data here
    dataFolderPath = 'images'  # path to images folder

    #Hyperparameters
    batchSize = 16
    learningRate = 1e-3
    numOfEpochs = 20
    valSize = 0.2
    rndSeed = 42

    
    #lImagePaths, lLabels = ListDataPaths(dataFolderPath) Not in use
    lImagePaths, lLabels, dDataStatistics = CalculateDatasetStatistics(dataFolderPath)

    positiveWeight = 1 #(len(lLabels) - np.sum(lLabels))/np.sum(lLabels)
    # 2. Perform stratified train-validation split
    lTrainImagePaths, lValImagePaths, lTrainLabels, lValLabels = train_test_split(lImagePaths, lLabels, test_size=valSize, stratify=lLabels, random_state=rndSeed)

    # 3. Create DataLoader instances for training and validation datasets
    transform = transforms.Compose([
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.RandomRotation(degrees=15),
                    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=dDataStatistics["imageMean"], std=dDataStatistics["imageStd"]),  # Image normalization
                ])

    paramsMean = dDataStatistics['paramsMean']
    paramStd   = dDataStatistics['paramsStd']

    tTrainDataset = EllipseDatasetClassification(list(lTrainImagePaths), list(lTrainLabels), paramsMean, paramStd, transform=transform)
    tValDataset   = EllipseDatasetClassification(list(lValImagePaths), list(lValLabels), paramsMean, paramStd, transform=transform)

   
        # Set device (GPU or CPU)
    mDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_counts = [len(lLabels)-np.sum(lLabels), np.sum(lLabels)]  # [Minority count (isEllipse = 0), Majority count (isEllipse = 1)]

    # Compute class weights (inverse frequency)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # Assign a weight to each sample based on its class
    sample_weights = [class_weights[int(label)] for _, label in tTrainDataset]

    # Create a sampler with replacement to oversample the minority class
    #sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    mTrainLoader = DataLoader(tTrainDataset, batch_size=batchSize, shuffle=True)
    mValLoader = DataLoader(tValDataset, batch_size=batchSize, shuffle=False)

    # Initialize model
    mModel = EllipseEstimatorModel().to(mDevice)
    mModel.apply(initialize_weights)
    positiveWeight = torch.tensor(1).to(mDevice) # positiveWeight

     # Train the model
    mModel = train_model(mModel, mTrainLoader, mValLoader, mDevice, positiveWeight, epochs=numOfEpochs)

