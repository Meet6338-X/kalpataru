"""
Image Processing Pipeline for Disease Detection.
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
from io import BytesIO

from config.settings import IMAGE_SIZE
from utils.logger import setup_logger

logger = setup_logger(__name__)

def process_uploaded_image(image_data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process uploaded image for disease detection.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Processed image array and metadata
    """
    image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Get metadata
    metadata = {
        'original_size': image.size,
        'format': image.format,
        'mode': image.mode
    }
    
    # Resize and normalize
    processed = resize_and_normalize(image, IMAGE_SIZE)
    
    return processed, metadata

def resize_and_normalize(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image and normalize pixel values.
    
    Args:
        image: PIL Image
        target_size: Target size (width, height)
        
    Returns:
        Normalized image array
    """
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return image_array

def apply_augmentation(image: np.ndarray) -> list:
    """
    Apply data augmentation for training.
    
    Args:
        image: Image array
        
    Returns:
        List of augmented images
    """
    augmented = [image]
    
    # Horizontal flip
    augmented.append(np.fliplr(image))
    
    # Rotation (simple 90 degree increments)
    for _ in range(1, 4):
        image = np.rot90(image)
        augmented.append(image)
    
    return augmented

def validate_image(image_data: bytes) -> bool:
    """
    Validate uploaded image.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        True if valid, False otherwise
    """
    try:
        image = Image.open(BytesIO(image_data))
        image.verify()
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False
