"""
Inference module for plant disease detection.
Supports 38 disease classes from PlantVillage dataset.
"""

from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np
from io import BytesIO
from pathlib import Path

from models.disease.cnn_model import (
    load_pretrained_model,
    load_pytorch_model,
    get_disease_info,
    get_all_disease_classes,
    DISEASE_CLASSES_38,
    DISEASE_INFO
)
from config.settings import DISEASE_MODEL_PATH, IMAGE_SIZE
from utils.logger import setup_logger
from services.explainability import generate_explanation

logger = setup_logger(__name__)

# Global model variable
_model = None
_model_type = None  # 'keras' or 'pytorch'


def get_model():
    """Get or load the disease detection model."""
    global _model, _model_type
    
    if _model is not None:
        return _model
    
    # Try loading Keras model first
    keras_path = DISEASE_MODEL_PATH
    if keras_path.exists():
        _model = load_pretrained_model(str(keras_path), num_classes=38)
        if _model is not None:
            _model_type = 'keras'
            logger.info("Disease detection Keras model loaded successfully")
            return _model
    
    # Try loading PyTorch model
    pytorch_path = Path(__file__).parent / 'plant-disease-model.pth'
    if pytorch_path.exists():
        _model = load_pytorch_model(str(pytorch_path), num_classes=38)
        if _model is not None:
            _model_type = 'pytorch'
            logger.info("Disease detection PyTorch model loaded successfully")
            return _model
    
    logger.warning("No disease detection model found. Using mock predictions.")
    _model_type = 'mock'
    return None


def preprocess_image(image_data: bytes, target_size: tuple = None) -> np.ndarray:
    """
    Preprocess image for disease detection.
    
    Args:
        image_data: Raw image bytes
        target_size: Target image size (default: IMAGE_SIZE from config)
        
    Returns:
        Preprocessed image array
    """
    if target_size is None:
        target_size = IMAGE_SIZE
    
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def preprocess_image_pytorch(image_data: bytes, target_size: tuple = (224, 224)):
    """
    Preprocess image for PyTorch model.
    
    Args:
        image_data: Raw image bytes
        target_size: Target image size
        
    Returns:
        Preprocessed tensor
    """
    try:
        import torch
        from torchvision import transforms
        
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
        
    except ImportError:
        logger.warning("PyTorch not available for preprocessing")
        return None


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
    
    if model is not None and _model_type == 'keras':
        return _predict_keras(model, image_data)
    elif model is not None and _model_type == 'pytorch':
        return _predict_pytorch(model, image_data)
    else:
        return _predict_mock(image_data)


def _predict_keras(model, image_data: bytes) -> Dict[str, Any]:
    """Make prediction using Keras model."""
    try:
        processed_image = preprocess_image(image_data)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        disease = DISEASE_CLASSES_38[predicted_class]
        
        return _build_result(disease, confidence, predictions[0])
        
    except Exception as e:
        logger.error(f"Keras prediction error: {str(e)}")
        return _predict_mock(image_data)


def _predict_pytorch(model, image_data: bytes) -> Dict[str, Any]:
    """Make prediction using PyTorch model."""
    try:
        import torch
        
        processed_image = preprocess_image_pytorch(image_data)
        
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        disease = DISEASE_CLASSES_38[predicted_class]
        
        # Convert to numpy for consistent interface
        all_probs = probabilities[0].numpy()
        
        return _build_result(disease, confidence, all_probs)
        
    except Exception as e:
        logger.error(f"PyTorch prediction error: {str(e)}")
        return _predict_mock(image_data)


def _predict_mock(image_data: bytes) -> Dict[str, Any]:
    """Generate mock prediction for development/testing."""
    # Try to analyze image color for basic prediction
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize((100, 100))
        pixels = np.array(image)
        
        # Analyze color distribution
        avg_color = pixels.mean(axis=(0, 1))
        r, g, b = avg_color
        
        # Simple heuristics based on color
        if g > r and g > b:
            # Greenish - likely healthy
            disease = 'Tomato___healthy'
            confidence = 0.85
        elif r > g and r > b:
            # Reddish - possible disease
            disease = 'Tomato___Early_blight'
            confidence = 0.70
        elif b > r and b > g:
            # Bluish - unusual
            disease = 'Tomato___Late_blight'
            confidence = 0.65
        else:
            # Default
            disease = 'Tomato___healthy'
            confidence = 0.60
            
    except Exception:
        disease = 'Tomato___healthy'
        confidence = 0.50
    
    return _build_result(disease, confidence, None)


def _build_result(disease: str, confidence: float, 
                  all_probs: Optional[np.ndarray]) -> Dict[str, Any]:
    """Build the result dictionary with disease information."""
    disease_info = get_disease_info(disease)
    
    result = {
        'disease': disease,
        'confidence': round(confidence, 4),
        'is_healthy': 'healthy' in disease.lower(),
        'description': disease_info['description'],
        'treatment': disease_info['treatment'],
        'prevention': disease_info['prevention'],
        'recommendations': get_recommendations(disease),
        'model_type': _model_type if _model_type else 'mock'
    }
    
    # Add top 5 predictions if available
    if all_probs is not None:
        top_indices = np.argsort(all_probs)[-5:][::-1]
        result['top_predictions'] = [
            {
                'disease': DISEASE_CLASSES_38[idx],
                'probability': round(float(all_probs[idx]), 4)
            }
            for idx in top_indices
        ]
    
    # Generate explanation
    result['explanation'] = generate_explanation(disease, confidence)
    
    return result


def get_recommendations(disease: str) -> List[str]:
    """Get treatment recommendations for the disease."""
    # Check if healthy
    if 'healthy' in disease.lower():
        return [
            'Continue regular plant care',
            'Monitor for any changes in plant health',
            'Maintain proper watering schedule',
            'Ensure adequate sunlight'
        ]
    
    # Disease-specific recommendations
    recommendations_map = {
        'Apple___Apple_scab': [
            'Apply fungicide (captan or mancozeb) during wet periods',
            'Remove and destroy fallen leaves',
            'Prune to improve air circulation',
            'Consider planting resistant varieties'
        ],
        'Apple___Black_rot': [
            'Prune out infected branches 8-12 inches below visible symptoms',
            'Remove mummified fruit from tree and ground',
            'Apply fungicide from bloom through fruit set',
            'Maintain tree vigor with proper fertilization'
        ],
        'Tomato___Bacterial_spot': [
            'Apply copper-based bactericide',
            'Remove infected leaves',
            'Avoid overhead irrigation',
            'Use disease-free seeds'
        ],
        'Tomato___Early_blight': [
            'Remove infected lower leaves',
            'Apply chlorothalonil or mancozeb fungicide',
            'Mulch around plants to prevent spore splash',
            'Ensure good air circulation'
        ],
        'Tomato___Late_blight': [
            'Remove and destroy infected plants immediately',
            'Apply preventive fungicide',
            'Avoid wetting foliage when watering',
            'Plant resistant varieties next season'
        ],
        'Tomato___Leaf_Mold': [
            'Increase ventilation in greenhouse',
            'Reduce humidity levels',
            'Apply fungicide if severe',
            'Space plants properly'
        ],
        'Tomato___Septoria_leaf_spot': [
            'Remove infected leaves',
            'Apply fungicide (chlorothalonil)',
            'Water at base of plants',
            'Rotate crops yearly'
        ],
        'Tomato___Spider_mites Two-spotted_spider_mite': [
            'Spray plants with water to dislodge mites',
            'Apply insecticidal soap or neem oil',
            'Introduce predatory mites',
            'Increase humidity around plants'
        ],
        'Tomato___Target_Spot': [
            'Apply fungicide (chlorothalonil or mancozeb)',
            'Improve air circulation',
            'Avoid overhead irrigation',
            'Remove infected plant debris'
        ],
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
            'Remove infected plants immediately',
            'Control whitefly population',
            'Use reflective mulches',
            'Plant resistant varieties'
        ],
        'Tomato___Tomato_mosaic_virus': [
            'Remove infected plants',
            'Disinfect tools and hands',
            'Control aphid vectors',
            'Use virus-free seeds'
        ],
        'Potato___Early_blight': [
            'Apply fungicide (chlorothalonil)',
            'Remove infected plant debris',
            'Practice crop rotation',
            'Ensure adequate plant spacing'
        ],
        'Potato___Late_blight': [
            'Apply preventive fungicide',
            'Destroy infected plants',
            'Ensure good drainage',
            'Use certified disease-free seed potatoes'
        ],
        'Grape___Black_rot': [
            'Apply fungicide every 7-14 days',
            'Remove mummified fruit',
            'Prune for good air circulation',
            'Destroy infected plant material'
        ],
        'Corn_(maize)___Common_rust_': [
            'Apply fungicide if infection is severe',
            'Plant resistant hybrids',
            'Avoid late planting',
            'Rotate crops'
        ]
    }
    
    # Get recommendations or default
    recommendations = recommendations_map.get(disease)
    
    if recommendations is None:
        # Try to find partial match
        for key in recommendations_map:
            if disease.split('___')[0] in key:
                recommendations = recommendations_map[key]
                break
    
    if recommendations is None:
        recommendations = [
            'Consult local agricultural extension service',
            'Remove infected plant material',
            'Apply appropriate treatment based on disease type',
            'Monitor plant health regularly'
        ]
    
    return recommendations


def get_supported_diseases() -> List[Dict[str, str]]:
    """
    Get list of all supported diseases with their information.
    
    Returns:
        List of dictionaries with disease names and descriptions
    """
    diseases = []
    for disease_name in DISEASE_CLASSES_38:
        info = get_disease_info(disease_name)
        diseases.append({
            'name': disease_name,
            'description': info['description'],
            'is_healthy': 'healthy' in disease_name.lower()
        })
    return diseases
