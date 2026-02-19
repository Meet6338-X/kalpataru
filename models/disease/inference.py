"""
Inference module for plant disease detection.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
from io import BytesIO

from models.disease.cnn_model import load_pretrained_model
from config.settings import DISEASE_MODEL_PATH, IMAGE_SIZE
from config.constants import DISEASE_CLASSES
from utils.logger import setup_logger
from services.explainability import generate_explanation

logger = setup_logger(__name__)

# Global model variable
_model = None

def get_model():
    """Get or load the disease detection model."""
    global _model
    if _model is None:
        try:
            _model = load_pretrained_model(str(DISEASE_MODEL_PATH))
            logger.info("Disease detection model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using mock predictions.")
            _model = None
    return _model

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess image for disease detection.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Preprocessed image array
    """
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_disease(image_file) -> Dict[str, Any]:
    """
    Predict disease from plant image.
    
    Args:
        image_file: Image file from request
        
    Returns:
        Prediction result dictionary
    """
    model = get_model()
    
    # Read image
    image_data = image_file.read()
    processed_image = preprocess_image(image_data)
    
    if model is not None:
        # Get prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        disease = DISEASE_CLASSES[predicted_class]
    else:
        # Mock prediction for development
        disease = "healthy"
        confidence = 0.95
    
    # Generate explanation
    explanation = generate_explanation(disease, confidence)
    
    return {
        'disease': disease,
        'confidence': confidence,
        'explanation': explanation,
        'recommendations': get_recommendations(disease)
    }

def get_recommendations(disease: str) -> list:
    """Get treatment recommendations for the disease."""
    recommendations = {
        'healthy': ['Continue regular plant care', 'Monitor for any changes'],
        'bacterial_spot': ['Remove infected leaves', 'Apply copper fungicide', 'Improve air circulation'],
        'early_blight': ['Remove infected leaves', 'Apply fungicide', 'Mulch around plants'],
        'late_blight': ['Remove infected plants', 'Apply fungicide', 'Ensure proper spacing'],
        'leaf_mold': ['Improve ventilation', 'Remove infected leaves', 'Reduce humidity'],
        'septoria_leaf_spot': ['Remove infected leaves', 'Apply fungicide', 'Water at base'],
        'spider_mites': ['Spray with water', 'Apply insecticidal soap', 'Increase humidity'],
        'target_spot': ['Remove infected leaves', 'Apply fungicide', 'Improve drainage'],
        'mosaic_virus': ['Remove infected plants', 'Control aphids', 'Use resistant varieties'],
        'yellow_leaf_curl_virus': ['Remove infected plants', 'Control whiteflies', 'Use resistant varieties']
    }
    return recommendations.get(disease, ['Consult local agricultural expert'])
