import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from PIL import Image
import numpy as np




# Dataset class 
class EllipseDatasetClassification(Dataset):
    def __init__(self, lImagePaths, lLabels, lParamsMean, lParamsSTD, transform=None):
        """
        Args:
            lImagePaths (list): List of image file paths.
            lLabels (list): List of labels (0 or 1) for classification.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.lImagePaths = lImagePaths
        self.lLabels     = lLabels
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
        label = self.lLabels[idx]
        tLabel = torch.tensor(label, dtype=torch.float32)

        # Apply any transformations to the image (e.g., normalization)
         # Read the image
        mImg = Image.open(imagePath).convert("RGB")  # Assuming image is stored in the specified path
        if self.transform:
            tImg = self.transform(mImg)

        #DEBUG    
        # pp = tImg.permute(1, 2, 0).numpy()
        # fig, ax=plt.subplots(1,1)
        # ax.imshow(pp)
        # fig.savefig("C:\\Users\\Naama\\Documents\\EllipseEstimator\\1.png")
        
        return tImg, tLabel