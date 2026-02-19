"""
Crop Recommendation Model.
"""

from typing import Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

def recommend_crop(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommend best crop for given conditions.
    
    Args:
        data: Input data with soil, climate, region info
        
    Returns:
        Crop recommendations
    """
    soil_type = data.get('soil_type', 'alluvial')
    rainfall = data.get('rainfall_mm', 1000)
    temperature = data.get('temperature_celsius', 25)
    ph_level = data.get('ph_level', 7.0)
    region = data.get('region', 'north')
    season = data.get('season', 'kharif')
    
    # Score each crop
    crops = get_all_crops()
    scored_crops = []
    
    for crop in crops:
        score = calculate_crop_score(crop, soil_type, rainfall, temperature, ph_level, season)
        scored_crops.append({
            'crop': crop['name'],
            'score': round(score, 2),
            'suitability': get_suitability_label(score),
            'reasons': get_reasons(crop, soil_type, rainfall, temperature, ph_level)
        })
    
    # Sort by score
    scored_crops.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'recommendations': scored_crops[:5],
        'best_crop': scored_crops[0]['crop'] if scored_crops else None,
        'input_conditions': {
            'soil_type': soil_type,
            'rainfall_mm': rainfall,
            'temperature_celsius': temperature,
            'ph_level': ph_level,
            'region': region,
            'season': season
        }
    }

def get_all_crops() -> List[Dict[str, Any]]:
    """Get all crop information."""
    return [
        {'name': 'rice', 'optimal_ph': [5.5, 7.0], 'optimal_temp': [20, 35], 'min_rainfall': 1000},
        {'name': 'wheat', 'optimal_ph': [6.0, 7.5], 'optimal_temp': [15, 25], 'min_rainfall': 500},
        {'name': 'maize', 'optimal_ph': [5.8, 7.0], 'optimal_temp': [20, 30], 'min_rainfall': 600},
        {'name': 'cotton', 'optimal_ph': [5.5, 8.0], 'optimal_temp': [25, 35], 'min_rainfall': 500},
        {'name': 'sugarcane', 'optimal_ph': [6.0, 8.0], 'optimal_temp': [24, 32], 'min_rainfall': 1500},
        {'name': 'potato', 'optimal_ph': [5.0, 6.5], 'optimal_temp': [15, 25], 'min_rainfall': 400},
        {'name': 'tomato', 'optimal_ph': [6.0, 6.8], 'optimal_temp': [20, 28], 'min_rainfall': 400},
        {'name': 'onion', 'optimal_ph': [6.0, 7.0], 'optimal_temp': [20, 30], 'min_rainfall': 350},
        {'name': 'mustard', 'optimal_ph': [6.0, 7.5], 'optimal_temp': [15, 25], 'min_rainfall': 400},
        {'name': 'soybean', 'optimal_ph': [5.5, 7.0], 'optimal_temp': [20, 30], 'min_rainfall': 450}
    ]

def calculate_crop_score(crop: Dict[str, Any], soil: str, rain: float, 
                        temp: float, ph: float, season: str) -> float:
    """Calculate suitability score for a crop."""
    score = 100.0
    
    # pH scoring
    ph_min, ph_max = crop['optimal_ph']
    if ph < ph_min:
        score -= (ph_min - ph) * 10
    elif ph > ph_max:
        score -= (ph - ph_max) * 10
    
    # Temperature scoring
    temp_min, temp_max = crop['optimal_temp']
    if temp < temp_min:
        score -= (temp_min - temp) * 3
    elif temp > temp_max:
        score -= (temp - temp_max) * 3
    
    # Rainfall scoring
    if rain < crop['min_rainfall']:
        score -= (crop['min_rainfall'] - rain) / 50
    
    # Soil type scoring
    soil_scores = {
        'alluvial': {'rice': 10, 'wheat': 10, 'sugarcane': 8, 'cotton': 7},
        'black': {'cotton': 10, 'sugarcane': 8, 'wheat': 6, 'maize': 7},
        'red': {'maize': 8, 'cotton': 7, 'groundnut': 8, 'ragi': 7},
        'laterite': {'tea': 10, 'coffee': 8, 'rubber': 7},
        'sandy': {'groundnut': 8, 'watermelon': 7, 'millet': 6}
    }
    soil_score = soil_scores.get(soil.lower(), {}).get(crop['name'], 0)
    score += soil_score
    
    return max(0, score)

def get_suitability_label(score: float) -> str:
    """Get suitability label from score."""
    if score >= 90:
        return 'excellent'
    elif score >= 75:
        return 'good'
    elif score >= 50:
        return 'moderate'
    else:
        return 'poor'

def get_reasons(crop: Dict, soil: str, rain: float, temp: float, ph: float) -> List[str]:
    """Get reasons for crop recommendation."""
    reasons = []
    
    ph_min, ph_max = crop['optimal_ph']
    if ph_min <= ph <= ph_max:
        reasons.append("Soil pH is within optimal range")
    else:
        reasons.append("pH adjustment may be needed")
    
    if rain >= crop['min_rainfall']:
        reasons.append("Adequate rainfall for this crop")
    else:
        reasons.append("Irrigation may be required")
    
    return reasons
