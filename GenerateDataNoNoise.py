import numpy as np
import cv2
import os
import random
import json

def generate_black_image(width, height):
    """
    Generate an image with a black background (all pixel values are 0).
    """
    return np.zeros((height, width, 3), dtype=np.uint8)  # Black RGB image

def generate_ellipse_parameters(width, height):
    """
    Generate random parameters for an ellipse.
    """
    center = (random.uniform(50, width - 50), random.uniform(50, height - 50))  # Avoid edges
    axes = (random.uniform(20, min(width, height) // 4), random.uniform(20, min(width, height) // 4))  # Random axes
    angle = random.uniform(0, 360)  # Random orientation
    color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color (RGB)
    thickness = random.randint(1, 5)  # Random thickness for contour
    return center, axes, angle, color, thickness

def add_contour_ellipse_to_image(image):
    """
    Add an ellipse contour with random parameters to the image and return metadata.
    """
    height, width, _ = image.shape
    center, axes, angle, color, thickness = generate_ellipse_parameters(width, height)
    cv2.ellipse(image, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle, 0, 360, color, thickness)
    
    metadata = {
        "isEllipse": True,
        "centerCoord": [center[0], center[1]],
        "radLength": [axes[0], axes[1]],
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
    0 - Black image (no ellipse)
    1 - Black image with ellipse contour
    """
    os.makedirs(output_dir, exist_ok=True)

    image_counter = 1

    for _ in range(num_class_0):
        # Generate black image for class 0
        black_image = generate_black_image(width, height)
        metadata = generate_metadata_for_class_0()

        # Generate file names
        base_filename = f"Image{image_counter:04d}"
        image_path = os.path.join(output_dir, f"{base_filename}.png")
        json_path = os.path.join(output_dir, f"{base_filename}.json")

        # Save the image and metadata
        cv2.imwrite(image_path, black_image)
        save_json(metadata, json_path)

        image_counter += 1

    for _ in range(num_class_1):
        # Generate black image with an ellipse contour for class 1
        black_image = generate_black_image(width, height)
        ellipse_image, metadata = add_contour_ellipse_to_image(black_image)

        # Generate file names
        base_filename = f"Image{image_counter:04d}"
        image_path = os.path.join(output_dir, f"{base_filename}.png")
        json_path = os.path.join(output_dir, f"{base_filename}.json")

        # Save the image and metadata
        cv2.imwrite(image_path, ellipse_image)
        save_json(metadata, json_path)

        image_counter += 1

    print(f"Dataset generated in {output_dir}")

# Parameters
output_dir = "ellipse_contour_data"  # Single folder for all images and JSON files
num_class_0 = 500  # Number of Class 0 images (black background)
num_class_1 = 500  # Number of Class 1 images (ellipse contour on black background)
width, height = 400, 400  # Image dimensions

# Generate the dataset
generate_dataset(output_dir, num_class_0, num_class_1, width, height)
