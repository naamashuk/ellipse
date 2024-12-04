import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json


directory = Path('Data/')
lImagePaths = [str(file) for file in directory.glob('*') if file.suffix.lower() in {'.png'}]

# List comprehension to extract the 'field_name' from each item in the JSON data
labels = []

for ii, imPath in enumerate(lImagePaths):
    print(ii)
    with open(imPath.replace('png', 'json'), 'r') as file:
        data = json.load(file)  # Parse the JSON data
        if data['isEllipse']:
            labels.append(1)
        else:
            labels.append(0)
    mImageRGB  = cv2.imread(imPath) 
    mImageGray = cv2.cvtColor(mImageRGB, cv2.COLOR_RGB2GRAY) 

    mean = np.mean(mImageGray)
    std = np.std(mImageGray)

    # Define a factor k to control the threshold sensitivity
    k = 2  # Adjust this factor based on your image content

    # Calculate adaptive lower and upper thresholds for Canny edge detection
    lower_threshold = max(0, mean - k * std)  # Ensure it's not negative
    upper_threshold = mean + k * std

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mImageGray, (7, 7), 1.5)

    thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Apply Canny edge detector with adaptive thresholds
    edges = cv2.Canny(blurred, int(lower_threshold), int(upper_threshold))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV thresholds for the ellipse contour
    # Adjust these thresholds based on the contour's expected color properties
    hue_min, hue_max = 0, 179         # Full hue range
    sat_min, sat_max = 50, 255        # High saturation to exclude gray
    val_min, val_max = 50, 255        # Exclude very dark regions

    # Create a mask for pixels within the defined HSV thresholds
    mask = cv2.inRange(hsv_image, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))

    # Refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Structuring element
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    # Extract the contour from the original image
    contour_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_cleaned)


    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the image
    axes[0].imshow(mImageGray, cmap='gray')
    axes[0].axis('off')  # Hide axes for the image

    # Plot the histogram
    axes[1].hist(mImageGray.ravel(), bins=256, color='black', alpha=0.7)
    axes[1].set_xlim(0, 255)  # Set x-axis range for pixel values (0-255)
    axes[1].set_title('Histogram')

    axes[2].imshow(thresh)
    axes[2].axis('off')  # Hide axes for the image

    savedFileName = f'C:\\Users\\Naama\\Documents\\ElipseEstimator\\debug\\image_{ii+1}.png'
    fig.savefig(savedFileName)
    # plt.show()
    


plt.imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray,(7,7),0)


plt.imshow(gray, cmap='gray')
plt.show()

a=1


