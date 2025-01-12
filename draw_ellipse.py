import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def generate_and_save_ellipse_comparison(image_size, gt_params, pred_params, save_dir, run_index, thickness=2, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """
    Generates an image with ground truth and predicted ellipses and saves it as a PNG file.
    
    Parameters:
    - image_size: Tuple specifying the size of the image (height, width)
    - gt_params: Ground truth parameters [x, y, a, b, theta]
    - pred_params: Predicted parameters [x, y, a, b, theta]
    - save_dir: Directory to save the PNG image
    - run_index: Index to name the file
    - thickness: Thickness of the ellipse contours (default: 2)
    - gt_color: Color of the ground truth ellipse (default: green)
    - pred_color: Color of the predicted ellipse (default: red)
    
    Returns:
    - file_path: Path to the saved PNG image
    """
    

    # Create a blank image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # Ground truth ellipse
    gt_center = (int(gt_params[0]), int(gt_params[1]))
    gt_axes = (int(gt_params[2]), int(gt_params[3]))
    gt_angle = np.degrees(gt_params[4])  # Convert radians to degrees
    cv2.ellipse(image, gt_center, gt_axes, gt_angle, 0, 360, gt_color, thickness)
    
    # Predicted ellipse
    pred_center = (int(pred_params[0]), int(pred_params[1]))
    pred_axes = (int(pred_params[2]), int(pred_params[3]))
    pred_angle = np.degrees(pred_params[4])  # Convert radians to degrees
    cv2.ellipse(image, pred_center, pred_axes, pred_angle, 0, 360, pred_color, thickness)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the image
    filename = f"ellipse_comparison_{run_index}.png"
    file_path = os.path.join(save_dir, filename)
    cv2.imwrite(file_path, image)
    
    print(f"Image saved to: {file_path}")
    return file_path

import cv2
import numpy as np


def draw_points_on_image(points, resolution, save_path, point_color=(255, 255, 255), point_radius=3, thickness=-1):
    """
    Draw points on a black image and save the image to a specified path.

    Args:
        points (list of tuple): List of (x, y) coordinates of points to draw.
        resolution (tuple): Resolution of the image as (width, height).
        save_path (str): Path to save the generated image.
        point_color (tuple): Color of the points in BGR format. Default is white.
        point_radius (int): Radius of the points. Default is 3.
        thickness (int): Thickness of the points. Default is -1 (filled circle).

    Returns:
        None
    """
    # Create a black image with the given resolution
    image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    
    # Draw each point on the image
    for point in points:
        cv2.circle(image, point, point_radius, point_color, thickness)
    
    # Save the image to the specified path
    cv2.imwrite(save_path, image)

# Example usage
points = [(50, 50), (100, 100), (200, 150)]  # Example points
resolution = (400, 300)  # Width, Height
save_path = "output_image.png"  # Save path

draw_points_on_image(points, resolution, save_path)


# Example usage
image_size = (400, 400)
gt_params = [200, 200, 100, 50, np.pi / 4]  # Ground truth parameters
pred_params = [200, 200, 90, 60, np.pi / 3]  # Predicted parameters
save_dir = "./ellipse_images"
run_index = 1

# Call the function
generate_and_save_ellipse_comparison(image_size, gt_params, pred_params, save_dir, run_index)

