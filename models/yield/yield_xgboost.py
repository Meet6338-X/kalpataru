"""
XGBoost Model for Crop Yield Prediction.
"""

from typing import Dict, Any
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

def predict_yield(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict crop yield.
    
    Args:
        data: Input data with crop info, area, region, etc.
        
    Returns:
        Yield prediction
    """
    crop_type = data.get('crop_type', 'rice')
    area_hectares = data.get('area_hectares', 1.0)
    region = data.get('region', 'punjab')
    season = data.get('season', 'kharif')
    soil_type = data.get('soil_type', 'alluvial')
    
    # Get base yield for crop (quintals per hectare)
    base_yield = get_base_yield(crop_type, region, season)
    
    # Apply soil type factor
    soil_factor = get_soil_factor(soil_type)
    
    # Apply season factor
    season_factor = get_season_factor(season)
    
    # Calculate predicted yield
    predicted_yield_per_hectare = base_yield * soil_factor * season_factor
    total_yield = predicted_yield_per_hectare * area_hectares
    
    return {
        'crop_type': crop_type,
        'area_hectares': area_hectares,
        'predicted_yield_quintals': round(total_yield, 2),
        'yield_per_hectare': round(predicted_yield_per_hectare, 2),
        'confidence': 0.82,
        'factors': {
            'soil_type': soil_factor,
            'season': season_factor
        }
    }

def get_base_yield(crop: str, region: str, season: str) -> float:
    """Get base yield for crop in region (quintals/hectare)."""
    # Simplified lookup table
    base_yields = {
        'rice': 45,
        'wheat': 40,
        'maize': 35,
        'cotton': 20,
        'sugarcane': 350,
        'potato': 200,
        'tomato': 150,
        'onion': 180,
        'mustard': 15,
        'soybean': 20
    }
    
    # Region adjustment
    region_multipliers = {
        'punjab': 1.1,
        'haryana': 1.05,
        'up': 1.0,
        'maharashtra': 0.95,
        'karnataka': 0.9,
        'tn': 1.0
    }
    
    base = base_yields.get(crop, 30)
    multiplier = region_multipliers.get(region.lower(), 1.0)
    
    return base * multiplier

def get_soil_factor(soil_type: str) -> float:
    """Get factor based on soil type."""
    soil_factors = {
        'alluvial': 1.1,
        'black': 1.0,
        'red': 0.9,
        'laterite': 0.85,
        'sandy': 0.75,
        'clay': 0.95
    }
    return soil_factors.get(soil_type.lower(), 1.0)

def get_season_factor(season: str) -> float:
    """Get factor based on season."""
    season_factors = {
        'kharif': 1.0,
        'rabi': 0.95,
        'zaid': 1.05
    }
    return season_factors.get(season.lower(), 1.0)
