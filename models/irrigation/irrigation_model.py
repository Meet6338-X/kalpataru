"""
Irrigation Model for water requirement prediction.
"""

from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)

def predict_irrigation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict irrigation requirements.
    
    Args:
        data: Input data including soil moisture, temperature, humidity, etc.
        
    Returns:
        Irrigation prediction result
    """
    soil_moisture = data.get('soil_moisture', 50)
    temperature = data.get('temperature', 25)
    humidity = data.get('humidity', 60)
    crop_type = data.get('crop_type', 'rice')
    growth_stage = data.get('growth_stage', 'vegetative')
    
    # Simplified calculation for demonstration
    # In production, this would use a trained ML model
    base_water_requirement = get_crop_water_requirement(crop_type, growth_stage)
    
    # Adjust based on conditions
    moisture_factor = max(0, (100 - soil_moisture) / 100)
    temp_factor = 1 + (max(0, temperature - 25) / 25) * 0.5
    humidity_factor = 1 + ((60 - humidity) / 60) * 0.2 if humidity < 60 else 1
    
    water_needed = base_water_requirement * moisture_factor * temp_factor * humidity_factor
    
    return {
        'water_requirement_mm': round(water_needed, 2),
        'irrigation_needed': water_needed > 5,
        'recommended_schedule': get_schedule(water_needed, crop_type),
        'confidence': 0.85
    }

def get_crop_water_requirement(crop_type: str, growth_stage: str) -> float:
    """Get base water requirement for crop stage (mm/day)."""
    water_requirements = {
        'rice': {'seedling': 5, 'vegetative': 7, 'flowering': 8, 'maturation': 5},
        'wheat': {'germination': 4, 'vegetative': 5, 'flowering': 6, 'maturation': 3},
        'maize': {'seedling': 4, 'vegetative': 6, 'flowering': 7, 'maturation': 4},
        'cotton': {'seedling': 3, 'vegetative': 5, 'flowering': 6, 'maturation': 4},
        'sugarcane': {'seedling': 4, 'vegetative': 6, 'flowering': 7, 'maturation': 5}
    }
    return water_requirements.get(crop_type, {}).get(growth_stage, 5)

def get_schedule(water_needed: float, crop_type: str) -> Dict[str, Any]:
    """Get recommended irrigation schedule."""
    if water_needed < 3:
        frequency = "Every 3-4 days"
    elif water_needed < 6:
        frequency = "Every 2 days"
    else:
        frequency = "Daily"
    
    return {
        'frequency': frequency,
        'best_time': 'Early morning (6-8 AM)',
        'duration_minutes': min(int(water_needed * 10), 60)
    }
