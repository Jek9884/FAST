import numpy as np

def apply_upsampling(image):
        
    # Dimensions of the original image
    original_height, original_width = image.shape

    # New dimensions
    new_height, new_width = 600, 600
    diff_height = new_height - original_height
    diff_width = new_width - original_width

    # Calculate the padding sizes for height and width
    top_pad = (new_height - original_height) // 2
    bottom_pad = new_height - original_height - top_pad
    left_pad = (new_width - original_width) // 2
    right_pad = new_width - original_width - left_pad
    
    background_color = 100

    # Create a new larger matrix filled with the padding value
    larger_matrix = np.full((new_height, new_width), background_color, dtype=image.dtype)
    
    # Copy the original image into the center of the larger matrix
    larger_matrix[top_pad:top_pad + original_height, left_pad:left_pad + original_width] = image

    return larger_matrix