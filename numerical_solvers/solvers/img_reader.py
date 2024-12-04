import cv2
import numpy as np

def read_img_in_grayscale(img_path, target_size=None):
    image = cv2.imread(img_path) 
    if target_size is not None: 
        image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    np_gray_image = np.array(gray_image)
    return np.float32(np_gray_image)


def standarize_grayscale_image_range(image) -> np.array:
    """
    Normalize the ixel values of a grayscale image to have a specified range [min_val, max_val].

    Parameters:
    image (np.ndarray): Grayscale image array, cv2 layout.
    Returns:
    np.ndarray: Normalized image array with pixel values in the range [min_val, max_val].
    """
    

    # Convert image to float32 to avoid issues with integer division
    image = image.astype(np.float32)
    
    # Compute the mean and standard deviation of the original image
    original_mean = np.mean(image)
    original_std = np.std(image)
    
    # Normalize the image to have mean 0 and std 1
    standardized_image = (image - original_mean) / original_std
    
    # Scale standardized image to fit in range [0, 1]
    min_std = np.min(standardized_image)
    max_std = np.max(standardized_image)
    scaled_image = (standardized_image - min_std) / (max_std - min_std)
        
    return scaled_image

def normalize_grayscale_image_range(image, min_val, max_val) -> np.array:
    """
    Normalize the ixel values of a grayscale image to have a specified range [min_val, max_val].

    Parameters:
    image (np.ndarray): Grayscale image array, cv2 layout.
    min_val (float): The minimum value of the desired range.
    max_val (float): The maximum value of the desired range.

    Returns:
    np.ndarray: Normalized image array with pixel values in the range [min_val, max_val].
    """
    

    # Convert image to float32 to avoid issues with integer division
    image = image.astype(np.float32)
    
    # Scale image to fit in range [0, 1]
    min_img = np.min(image)
    max_img = np.max(image)
    normalized_image = (image - min_img) / (max_img - min_img)
    
    # Scale to the desired range [min_val, max_val]
    scaled_image = normalized_image * (max_val - min_val) + min_val
    return scaled_image

def change_value_range(image, prev_min_val, prev_max_val, min_val, max_val) -> np.array:
    image = image.astype(np.float32)
    return min_val + (image - prev_min_val) * (max_val - min_val) / (prev_max_val - prev_min_val)
