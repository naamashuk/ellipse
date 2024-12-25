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

# Example usage
image_size = (400, 400)
gt_params = [200, 200, 100, 50, np.pi / 4]  # Ground truth parameters
pred_params = [200, 200, 90, 60, np.pi / 3]  # Predicted parameters
save_dir = "./ellipse_images"
run_index = 1

# Call the function
generate_and_save_ellipse_comparison(image_size, gt_params, pred_params, save_dir, run_index)

