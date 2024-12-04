import numpy as np
import cv2
import os
import json

def generate_noisy_image_rgb(width, height, noise_type='gaussian'):
    """
    Generate a noisy RGB image with a 50% gray background.
    
    Args:
        width (int): Image width.
        height (int): Image height.
        noise_type (str): Type of noise ('gaussian' or 'uniform').
        
    Returns:
        np.ndarray: Noisy RGB image.
    """
    # Create a 50% gray background
    gray_background = np.full((height, width, 3), 128, dtype=np.uint8)  # Three channels for RGB
    
    # Add noise to each channel independently
    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0, scale=25, size=(height, width, 3))  # Gaussian noise
    elif noise_type == 'uniform':
        noise = np.random.uniform(-25, 25, size=(height, width, 3))  # Uniform noise
    else:
        raise ValueError("Invalid noise_type. Use 'gaussian' or 'uniform'.")

    # Clip values to stay within valid range [0, 255]
    noisy_image = gray_background + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def generate_metadata_json(output_path):
    """
    Generate a JSON file with metadata for the `0` class.
    
    Args:
        output_path (str): Path to save the JSON file.
    """
    metadata = {
        "isEllipse": False,
        "centerCoord": [0, 0],  # Placeholder
        "radLength": [0, 0],   # Placeholder
        "rotAngle": 0.0        # Placeholder
    }
    
    with open(output_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

def save_images_and_metadata(output_dir, num_images, width, height, noise_type='gaussian'):
    """
    Generate and save noisy RGB images along with corresponding JSON metadata files.
    
    Args:
        output_dir (str): Directory to save the images and JSON files.
        num_images (int): Number of images to generate.
        width (int): Image width.
        height (int): Image height.
        noise_type (str): Type of noise ('gaussian' or 'uniform').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Generate the noisy image
        noisy_image = generate_noisy_image_rgb(width, height, noise_type)
        base_filename = f"ImageGen{i:04d}.png"  # Pads numbers to 4 digits, e.g., ImageGen0001
        image_path = os.path.join(output_dir, base_filename)
        cv2.imwrite(image_path, noisy_image)
        
        # Generate corresponding JSON metadata
        json_filename = base_filename.replace('.png', '.json')
        json_path = os.path.join(output_dir, json_filename)
        generate_metadata_json(json_path)
        
        print(f"Saved: {image_path} and {json_path}")

# Generate 50 noisy RGB images of size 400x400 and corresponding JSON files
output_dir = "noisy_rgb_dataset"
save_images_and_metadata(output_dir, num_images=60, width=400, height=400, noise_type='gaussian')

