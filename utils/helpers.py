"""
Helper utilities for Kalpataru application.
"""

import base64
import io
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def preprocess_image(image: Image.Image, size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image object
        size: Target image size
        
    Returns:
        Preprocessed image as numpy array
    """
    image = image.resize(size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def format_prediction_result(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format prediction result for API response.
    
    Args:
        prediction: Raw prediction dictionary
        
    Returns:
        Formatted prediction dictionary
    """
    return {
        'prediction': prediction.get('class', 'unknown'),
        'confidence': float(prediction.get('probability', 0.0)),
        'timestamp': prediction.get('timestamp', None)
    }

def validate_image_file(filename: str) -> bool:
    """
    Validate if the file is a valid image.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if valid image file, False otherwise
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def create_response(success: bool, data: Any = None, message: str = '') -> Dict[str, Any]:
    """
    Create standardized API response.
    
    Args:
        success: Whether operation was successful
        data: Response data
        message: Optional message
        
    Returns:
        Standardized response dictionary
    """
    response = {'success': success}
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    return response
