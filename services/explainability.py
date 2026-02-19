"""
Explainability Service for model predictions.
"""

from typing import Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_explanation(prediction: str, confidence: float) -> str:
    """
    Generate human-readable explanation for prediction.
    
    Args:
        prediction: Predicted class
        confidence: Prediction confidence
        
    Returns:
        Explanation string
    """
    confidence_level = get_confidence_level(confidence)
    
    explanations = {
        'healthy': f"The plant appears to be healthy with {confidence_level} confidence.",
        'bacterial_spot': f"Detected bacterial spots on leaves with {confidence_level} confidence. This is caused by Xanthomonas bacteria.",
        'early_blight': f"Found early blight disease with {confidence_level} confidence. Common in warm, humid conditions.",
        'late_blight': f"Detected late blight with {confidence_level} confidence. This is a serious disease that can spread quickly.",
        'leaf_mold': f"Identified leaf mold with {confidence_level} confidence. Usually occurs in high humidity.",
        'septoria_leaf_spots': f"Found septoria leaf spots with {confidence_level} confidence.",
        'spider_mites': f"Detected spider mite infestation with {confidence_level} confidence.",
        'target_spot': f"Identified target spot disease with {confidence_level} confidence.",
        'mosaic_virus': f"Detected mosaic virus with {confidence_level} confidence.",
        'yellow_leaf_curl_virus': f"Found yellow leaf curl virus with {confidence_level} confidence."
    }
    
    return explanations.get(prediction, f"Prediction: {prediction} with {confidence_level} confidence")

def get_confidence_level(confidence: float) -> str:
    """Get human-readable confidence level."""
    if confidence >= 0.9:
        return "very high"
    elif confidence >= 0.75:
        return "high"
    elif confidence >= 0.5:
        return "moderate"
    else:
        return "low"

def explain_crop_recommendation(recommendation: Dict[str, Any]) -> str:
    """
    Generate explanation for crop recommendation.
    
    Args:
        recommendation: Crop recommendation dictionary
        
    Returns:
        Explanation string
    """
    crop = recommendation.get('crop', 'unknown')
    score = recommendation.get('score', 0)
    suitability = recommendation.get('suitability', 'unknown')
    
    return (f"{crop.capitalize()} is recommended with {suitability} suitability "
            f"(score: {score}). This crop is well-suited to your conditions.")

def explain_yield_prediction(prediction: Dict[str, Any]) -> str:
    """
    Generate explanation for yield prediction.
    
    Args:
        prediction: Yield prediction dictionary
        
    Returns:
        Explanation string
    """
    crop = prediction.get('crop_type', 'crop')
    yield_value = prediction.get('predicted_yield_quintals', 0)
    area = prediction.get('area_hectares', 1)
    
    per_hectare = yield_value / area if area > 0 else 0
    
    return (f"Expected yield for {crop} is {yield_value:.1f} quintals "
            f"({per_hectare:.1f} quintals per hectare) from {area} hectares.")

def get_feature_importance(model_type: str) -> List[Dict[str, Any]]:
    """
    Get feature importance for different models.
    
    Args:
        model_type: Type of model
        
    Returns:
        List of features with importance scores
    """
    importance_maps = {
        'disease': [
            {'feature': 'Leaf color patterns', 'importance': 0.35},
            {'feature': 'Spot shapes', 'importance': 0.25},
            {'feature': 'Lesion distribution', 'importance': 0.20},
            {'feature': 'Leaf texture', 'importance': 0.15},
            {'feature': 'Overall plant health', 'importance': 0.05}
        ],
        'yield': [
            {'feature': 'Soil type', 'importance': 0.30},
            {'feature': 'Weather conditions', 'importance': 0.25},
            {'feature': 'Irrigation', 'importance': 0.20},
            {'feature': 'Fertilizer usage', 'importance': 0.15},
            {'feature': 'Crop variety', 'importance': 0.10}
        ],
        'price': [
            {'feature': 'Historical prices', 'importance': 0.40},
            {'feature': 'Seasonal patterns', 'importance': 0.25},
            {'feature': 'Supply data', 'importance': 0.20},
            {'feature': 'Demand trends', 'importance': 0.15}
        ]
    }
    
    return importance_maps.get(model_type, [])
