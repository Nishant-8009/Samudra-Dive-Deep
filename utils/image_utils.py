import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from config import TEMP_DIR

def save_uploaded_image(image, filename):
    """
    Save the uploaded image to a temporary directory.
    
    Args:
        image (PIL.Image): Image to save
        filename (str): Original filename
        
    Returns:
        str: Path to the saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Get file extension
    ext = os.path.splitext(filename)[1].lower()
    
    # Save image to temp directory
    path = os.path.join(TEMP_DIR, f"input{ext}")
    image.save(path)
    
    return path
