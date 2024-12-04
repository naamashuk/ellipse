import torch
import torch.nn as nn
import torch.nn.init as init



class EllipseEstimatorModel_new(nn.Module):
    def __init__(self):
        super(EllipseEstimatorModel_new, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Reduced to 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced to 32 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced to 64 channels
        self.conv4 = nn.Conv2d(64, 4, kernel_size=3, padding=1)  # Reduced to 64 channels
        self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by half
        
        # Global Average Pooling reduces spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(10000, 1024)  # Reduced to 64 units
        self.fc2 = nn.Linear(1024, 512)  # Output for binary classification
        self.fc3 = nn.Linear(512, 1)  # Output for regression parameters

    def forward(self, mImg):
        # Apply convolutional and pooling layers
        mImg = self.pool(torch.relu(self.conv1(mImg)))
        mImg = self.pool(torch.relu(self.conv2(mImg)))
        mImg = self.pool(torch.relu(self.conv3(mImg)))
        mImg = torch.relu(self.conv4(mImg))


        
        # Apply global average pooling
        #mImg = self.global_pool(mImg)  # Shape: [batch_size, 64, 1, 1]
        mImg = mImg.view(mImg.size(0), -1)  # Flatten to [batch_size, 64]

        # Apply fully connected layers
        mImg = torch.relu(self.fc1(mImg))
        mImg = torch.relu(self.fc2(mImg))


        mClassOutput = torch.sigmoid(self.fc3(mImg))  # Sigmoid for binary classification
        #mRegOutput = self.fc3(mImg)  # Regression for ellipse parameters
        
        return mClassOutput

class EllipseEstimatorModel(nn.Module):
    def __init__(self):
        super(EllipseEstimatorModel, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Reduced to 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced to 32 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced to 64 channels
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Reduced to 64 channels
        self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by half
        
        # Global Average Pooling reduces spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 64)  # Reduced to 64 units
        self.fc2 = nn.Linear(64, 1)  # Output for binary classification
        self.fc3 = nn.Linear(64, 5)  # Output for regression parameters

    def forward(self, mImg):
        # Apply convolutional and pooling layers
        #print(mImg.shape)
        mImg = self.pool(torch.relu(self.conv1(mImg)))
        #print(mImg.shape)
        mImg = self.pool(torch.relu(self.conv2(mImg)))
        #print(mImg.shape)
        mImg = self.pool(torch.relu(self.conv3(mImg)))
        #print(mImg.shape)

        mImg = self.pool(torch.relu(self.conv4(mImg)))
        #print(mImg.shape)
        
        # Apply global average pooling
        mImg = self.global_pool(mImg)  # Shape: [batch_size, 64, 1, 1]
        #print(mImg.shape)

        mImg = mImg.view(mImg.size(0), -1)  # Flatten to [batch_size, 64]

        # Apply fully connected layers
        mImg = torch.relu(self.fc1(mImg))
        mClassOutput = torch.sigmoid(self.fc2(mImg))  # Sigmoid for binary classification
        #mRegOutput = self.fc3(mImg)  # Regression for ellipse parameters
        
        return mClassOutput #, mRegOutput
    
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)

    
# # Define the CNN model
# class EllipseEstimatorModel_old(nn.Module):
#     def __init__(self):
#         super(EllipseEstimatorModel, self).__init__()
        
#         # Define layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by half
        
#         # Calculate flattened size after conv and pooling
#         self.flattened_size = 64 * (400 // 4) * (400 // 4)  # (400 is input size, //4 accounts for 2 pooling layers)

#         # Fully connected layers
#         self.fc1 = nn.Linear(self.flattened_size, 128)
#         self.fc2 = nn.Linear(128, 1)  # Output for binary classification
#         self.fc3 = nn.Linear(128, 5)  # Output for regression parameters

#     def forward(self, mImg):class EllipseEstimatorModel_old(nn.Module):
#     def __init__(self):
#         super(EllipseEstimatorModel_old, self).__init__()
        
#         # Define convolutional layers
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Reduced to 16 channels
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced to 32 channels
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced to 64 channels
#         self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by half
        
#         # Global Average Pooling reduces spatial dimensions to 1x1
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(64, 64)  # Reduced to 64 units
#         self.fc2 = nn.Linear(64, 1)  # Output for binary classification
#         self.fc3 = nn.Linear(64, 5)  # Output for regression parameters

#     def forward(self, mImg):
#         # Apply convolutional and pooling layers
#         mImg = self.pool(torch.relu(self.conv1(mImg)))
#         mImg = self.pool(torch.relu(self.conv2(mImg)))
#         mImg = self.pool(torch.relu(self.conv3(mImg)))
        
#         # Apply global average pooling
#         mImg = self.global_pool(mImg)  # Shape: [batch_size, 64, 1, 1]
#         mImg = mImg.view(mImg.size(0), -1)  # Flatten to [batch_size, 64]

#         # Apply fully connected layers
#         mImg = torch.relu(self.fc1(mImg))
#         mClassOutput = torch.sigmoid(self.fc2(mImg))  # Sigmoid for binary classification
#         mRegOutput = self.fc3(mImg)  # Regression for ellipse parameters
        
#         return mClassOutput, mRegOutput
#         mImg = self.pool(torch.relu(self.conv1(mImg)))
#         mImg = self.pool(torch.relu(self.conv2(mImg)))
#         mImg = mImg.view(mImg.size(0), -1)  # Flatten

#         mImg = self.dropout(torch.relu(self.fc1(mImg)))
#         mClassOutput = torch.sigmoid(self.fc2(mImg))  # Sigmoid for binary classification
#         mRegOutput = self.fc3(mImg)  # Regression for ellipse parameters
#         return mClassOutput, mRegOutput