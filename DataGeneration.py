import numpy as np
import cv2
import os
import random
import json

def generate_white_noise_image(height=400, width=400, mean=0, std=1):
    """
    Generates an RGB image with white noise.
    
    Parameters:
        height (int): Height of the image.
        width (int): Width of the image.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        np.ndarray: Generated white noise RGB image.
    """
    # Generate white noise for each channel
    noise_r = np.random.normal(mean, std, (height, width))
    noise_g = np.random.normal(mean, std, (height, width))
    noise_b = np.random.normal(mean, std, (height, width))
    
    # Stack channels to create an RGB image
    noise_rgb = np.stack((noise_r, noise_g, noise_b), axis=-1)
    
    # Normalize the values to 0-255 for visualization
    noise_rgb_normalized = (noise_rgb - noise_rgb.min()) / (noise_rgb.max() - noise_rgb.min()) * 255
    noise_rgb_normalized = noise_rgb_normalized.astype(np.uint8)
    
    return noise_rgb_normalized

def generate_zeros_image(width, height):
    """
    Generate an image with zeros as background.
    """
    image = np.zeros((height, width, 3))
    return image

def generate_ellipse_parameters(width, height):
    """
    Generate random parameters for an ellipse.
    """
    center = (random.uniform(50, width - 50), random.uniform(50, height - 50))  # Avoid edges
    axes = (random.uniform(20, min(width, height) // 4), random.uniform(20, min(width, height) // 4))  # Random axes
    angle = random.uniform(0, 360)  # Random orientation
    color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color (RGB)
    return center, axes, angle, color

def add_filled_ellipse_to_image(image):
    """
    Add a filled ellipse with random parameters to the image and return metadata.
    """
    height, width, _ = image.shape
    center, axes, angle, color = generate_ellipse_parameters(width, height)
    cv2.ellipse(image, (int(center[0]), int(center[1])), (int(np.max(axes)), int(np.min(axes))),
                angle, 0, 360, color, thickness=2)  # Use thickness=-1 for a filled ellipse
    
    metadata = {
        "isEllipse": True,
        "centerCoord": [int(center[0]), int(center[1])],
        "radLength": [int(max(axes[0], axes[1])), int(min(axes[0], axes[1]))],
        "rotAngle": np.deg2rad(angle)  # Convert to radians
    }
    return image, metadata

def generate_metadata_for_class_0():
    """
    Generate metadata for class 0 (no ellipse).
    """
    return {
        "isEllipse": False,
        "centerCoord": [0.0, 0.0],
        "radLength": [0.0, 0.0],
        "rotAngle": 0.0
    }

def save_json(metadata, output_path):
    """
    Save metadata as a JSON file.
    """
    with open(output_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

def generate_dataset(output_dir, num_class_0, num_class_1, width, height):
    """
    Generate a dataset with two classes in a single folder:
    0 - White noise only
    1 - White noise with a filled ellipse
    """
    os.makedirs(output_dir, exist_ok=True)

    image_counter = 1

    for _ in range(num_class_0):
        # Generate white noise image for class 0
        noise_image = generate_white_noise_image() #generate_zeros_image(width, height) #
        metadata = generate_metadata_for_class_0()

        # Generate file names
        base_filename = f"Image{image_counter:04d}"
        image_path = os.path.join(output_dir, f"{base_filename}.png")
        json_path = os.path.join(output_dir, f"{base_filename}.json")

        # Save the image and metadata
        cv2.imwrite(image_path, noise_image)
        save_json(metadata, json_path)

        image_counter += 1

    for _ in range(num_class_1):
        # Generate white noise image with a filled ellipse for class 1
        noise_image = generate_white_noise_image() #generate_zeros_image(width, height) #
        ellipse_image, metadata = add_filled_ellipse_to_image(noise_image)

        # Generate file names
        if metadata['radLength'][1] > metadata['radLength'][0]:
            u=1
        base_filename = f"Image{image_counter:04d}"
        image_path = os.path.join(output_dir, f"{base_filename}.png")
        json_path = os.path.join(output_dir, f"{base_filename}.json")

        # Save the image and metadata
        cv2.imwrite(image_path, ellipse_image)
        save_json(metadata, json_path)

        image_counter += 1

    print(f"Dataset generated in {output_dir}")

# Parameters
output_dir = "ellipse_data2"  # Single folder for all images and JSON files
num_class_0 = 500  # Number of Class 0 images (no ellipse)
num_class_1 = 500  # Number of Class 1 images (with filled ellipse)
width, height = 400, 400  # Image dimensions

# Generate the dataset
generate_dataset(output_dir, num_class_0, num_class_1, width, height)
