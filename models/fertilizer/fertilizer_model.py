"""
Fertilizer Recommendation Model for Kalpataru.
ML-based fertilizer recommendations based on soil nutrients, crop type, and environmental conditions.
"""

from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Global model variables
_model = None
_soil_encoder = None
_crop_encoder = None
_fertilizer_encoder = None


def get_model_path() -> Path:
    """Get the path to the fertilizer model directory."""
    return Path(__file__).parent


def load_models():
    """Load the trained fertilizer recommendation model and encoders."""
    global _model, _soil_encoder, _crop_encoder, _fertilizer_encoder
    
    if _model is not None:
        return _model, _soil_encoder, _crop_encoder, _fertilizer_encoder
    
    model_dir = get_model_path()
    
    try:
        model_path = model_dir / 'fertilizer_model.pkl'
        soil_encoder_path = model_dir / 'soil_encoder.pkl'
        crop_encoder_path = model_dir / 'crop_encoder.pkl'
        fertilizer_encoder_path = model_dir / 'fertilizer_encoder.pkl'
        
        if model_path.exists():
            _model = joblib.load(model_path)
            _soil_encoder = joblib.load(soil_encoder_path)
            _crop_encoder = joblib.load(crop_encoder_path)
            _fertilizer_encoder = joblib.load(fertilizer_encoder_path)
            logger.info("Fertilizer recommendation model loaded successfully")
        else:
            logger.warning("Fertilizer model files not found. Using rule-based predictions.")
            
    except Exception as e:
        logger.error(f"Error loading fertilizer model: {str(e)}")
        
    return _model, _soil_encoder, _crop_encoder, _fertilizer_encoder


class FertilizerRecommender:
    """
    Fertilizer recommendation class that combines ML model predictions
    with rule-based fallback for robust recommendations.
    """
    
    # Soil types supported
    SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    
    # Crop types supported
    CROP_TYPES = [
        'Wheat', 'Cotton', 'Maize', 'Paddy', 'Barley', 
        'Ground Nuts', 'Sugarcane', 'Potato', 'Tomato', 'Onion'
    ]
    
    # Fertilizer database with NPK ratios and application info
    FERTILIZER_DB = {
        'Urea': {
            'npk': (46, 0, 0),
            'type': 'Nitrogen',
            'application': 'Apply in split doses during growing season',
            'notes': 'Best applied when soil is moist'
        },
        'DAP': {
            'npk': (18, 46, 0),
            'type': 'Phosphorus-Nitrogen',
            'application': 'Apply at sowing/transplanting',
            'notes': 'Avoid direct contact with seeds'
        },
        'MOP': {
            'npk': (0, 0, 60),
            'type': 'Potassium',
            'application': 'Apply during active growth phase',
            'notes': 'Good for fruit development'
        },
        'NPK-10-26-26': {
            'npk': (10, 26, 26),
            'type': 'Balanced',
            'application': 'Apply at flowering stage',
            'notes': 'Good for root crops'
        },
        'NPK-20-20-20': {
            'npk': (20, 20, 20),
            'type': 'Balanced',
            'application': 'Apply during vegetative growth',
            'notes': 'Suitable for most crops'
        },
        'SSP': {
            'npk': (0, 16, 0),
            'type': 'Phosphorus',
            'application': 'Apply before sowing',
            'notes': 'Contains sulfur and calcium'
        },
        'Ammonium Sulphate': {
            'npk': (20.6, 0, 0),
            'type': 'Nitrogen-Sulfur',
            'application': 'Apply in acidic soils',
            'notes': 'Good for tea and vegetables'
        },
        'Super Phosphate': {
            'npk': (0, 36, 0),
            'type': 'Phosphorus',
            'application': 'Apply at planting time',
            'notes': 'Promotes root development'
        },
        '10-10-10': {
            'npk': (10, 10, 10),
            'type': 'Balanced',
            'application': 'General purpose fertilizer',
            'notes': 'Suitable for beginners'
        },
        '14-35-14': {
            'npk': (14, 35, 14),
            'type': 'Phosphorus-rich',
            'application': 'Apply at early growth stage',
            'notes': 'Good for legumes'
        }
    }
    
    # Crop-specific nutrient requirements (kg/ha)
    CROP_NPK_REQUIREMENTS = {
        'Wheat': {'N': 120, 'P': 60, 'K': 40},
        'Cotton': {'N': 90, 'P': 45, 'K': 45},
        'Maize': {'N': 150, 'P': 75, 'K': 60},
        'Paddy': {'N': 100, 'P': 50, 'K': 50},
        'Barley': {'N': 80, 'P': 40, 'K': 30},
        'Ground Nuts': {'N': 20, 'P': 40, 'K': 40},
        'Sugarcane': {'N': 250, 'P': 125, 'K': 100},
        'Potato': {'N': 150, 'P': 100, 'K': 150},
        'Tomato': {'N': 100, 'P': 80, 'K': 80},
        'Onion': {'N': 100, 'P': 60, 'K': 60}
    }
    
    def __init__(self):
        """Initialize the fertilizer recommender."""
        self.model, self.soil_encoder, self.crop_encoder, self.fertilizer_encoder = load_models()
    
    def predict(self, temperature: float, humidity: float, moisture: float,
                soil_type: str, crop_type: str, nitrogen: int, 
                potassium: int, phosphorous: int) -> Dict[str, Any]:
        """
        Predict the best fertilizer for given conditions.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            moisture: Soil moisture percentage
            soil_type: Type of soil (Sandy, Loamy, Black, Red, Clayey)
            crop_type: Type of crop
            nitrogen: Nitrogen level in soil
            potassium: Potassium level in soil
            phosphorous: Phosphorous level in soil
            
        Returns:
            Dictionary with fertilizer recommendation and details
        """
        # Try ML model first
        if self.model is not None:
            try:
                result = self._predict_ml(
                    temperature, humidity, moisture, soil_type, 
                    crop_type, nitrogen, potassium, phosphorous
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"ML prediction failed: {str(e)}. Falling back to rule-based.")
        
        # Fallback to rule-based prediction
        return self._predict_rule_based(
            temperature, humidity, moisture, soil_type,
            crop_type, nitrogen, potassium, phosphorous
        )
    
    def _predict_ml(self, temperature: float, humidity: float, moisture: float,
                    soil_type: str, crop_type: str, nitrogen: int,
                    potassium: int, phosphorous: int) -> Optional[Dict[str, Any]]:
        """Use ML model for prediction."""
        try:
            # Encode categorical inputs
            soil_encoded = self.soil_encoder.transform([soil_type])[0]
            crop_encoded = self.crop_encoder.transform([crop_type])[0]
            
            # Create input DataFrame with correct feature names
            input_data = pd.DataFrame([[
                temperature, humidity, moisture,
                soil_encoded, crop_encoded,
                nitrogen, potassium, phosphorous
            ]], columns=[
                'Temparature', 'Humidity ', 'Moisture',
                'Soil Type', 'Crop Type',
                'Nitrogen', 'Potassium', 'Phosphorous'
            ])
            
            # Predict
            pred_encoded = self.model.predict(input_data)[0]
            fertilizer = self.fertilizer_encoder.inverse_transform([pred_encoded])[0]
            
            return self._build_result(fertilizer, nitrogen, phosphorous, potassium, crop_type)
            
        except Exception as e:
            logger.error(f"ML prediction error: {str(e)}")
            return None
    
    def _predict_rule_based(self, temperature: float, humidity: float, moisture: float,
                            soil_type: str, crop_type: str, nitrogen: int,
                            potassium: int, phosphorous: int) -> Dict[str, Any]:
        """Rule-based fertilizer recommendation."""
        # Calculate nutrient gaps
        crop_needs = self.CROP_NPK_REQUIREMENTS.get(crop_type, {'N': 100, 'P': 50, 'K': 50})
        
        n_gap = crop_needs['N'] - nitrogen
        p_gap = crop_needs['P'] - phosphorous
        k_gap = crop_needs['K'] - potassium
        
        # Determine primary deficiency
        gaps = {'N': n_gap, 'P': p_gap, 'K': k_gap}
        max_deficiency = max(gaps, key=gaps.get)
        
        # Select fertilizer based on deficiency
        if gaps[max_deficiency] <= 0:
            # No major deficiency - use balanced fertilizer
            fertilizer = 'NPK-20-20-20'
        elif max_deficiency == 'N':
            if p_gap > 20:
                fertilizer = 'DAP'
            else:
                fertilizer = 'Urea'
        elif max_deficiency == 'P':
            if n_gap > 20:
                fertilizer = 'DAP'
            else:
                fertilizer = 'SSP'
        else:  # K deficiency
            fertilizer = 'MOP'
        
        # Adjust for soil type
        if soil_type.lower() == 'sandy' and moisture < 30:
            if fertilizer == 'Urea':
                fertilizer = 'Ammonium Sulphate'  # Better for sandy soils
        
        return self._build_result(fertilizer, nitrogen, phosphorous, potassium, crop_type)
    
    def _build_result(self, fertilizer: str, nitrogen: int, phosphorous: int, 
                      potassium: int, crop_type: str) -> Dict[str, Any]:
        """Build the result dictionary with fertilizer details."""
        fert_info = self.FERTILIZER_DB.get(fertilizer, {
            'npk': (0, 0, 0),
            'type': 'Unknown',
            'application': 'Follow package instructions',
            'notes': 'Consult local agricultural expert'
        })
        
        crop_needs = self.CROP_NPK_REQUIREMENTS.get(crop_type, {'N': 100, 'P': 50, 'K': 50})
        
        return {
            'recommended_fertilizer': fertilizer,
            'fertilizer_type': fert_info['type'],
            'npk_ratio': {
                'nitrogen': fert_info['npk'][0],
                'phosphorus': fert_info['npk'][1],
                'potassium': fert_info['npk'][2]
            },
            'application_method': fert_info['application'],
            'application_notes': fert_info['notes'],
            'nutrient_analysis': {
                'current_npk': {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorous,
                    'potassium': potassium
                },
                'recommended_npk': crop_needs,
                'gaps': {
                    'nitrogen': max(0, crop_needs['N'] - nitrogen),
                    'phosphorus': max(0, crop_needs['P'] - phosphorous),
                    'potassium': max(0, crop_needs['K'] - potassium)
                }
            },
            'confidence': 0.85 if self.model else 0.70
        }


def predict_fertilizer(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to predict fertilizer recommendation.
    
    Args:
        data: Dictionary containing input parameters:
            - temperature: float (°C)
            - humidity: float (%)
            - moisture: float (%)
            - soil_type: str
            - crop_type: str
            - nitrogen: int
            - potassium: int
            - phosphorous: int
            
    Returns:
        Fertilizer recommendation dictionary
    """
    recommender = FertilizerRecommender()
    
    return recommender.predict(
        temperature=data.get('temperature', 25.0),
        humidity=data.get('humidity', 60.0),
        moisture=data.get('moisture', 50.0),
        soil_type=data.get('soil_type', 'Loamy'),
        crop_type=data.get('crop_type', 'Wheat'),
        nitrogen=data.get('nitrogen', 50),
        potassium=data.get('potassium', 40),
        phosphorous=data.get('phosphorous', 30)
    )


def get_fertilizer_info(fertilizer_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific fertilizer.
    
    Args:
        fertilizer_name: Name of the fertilizer
        
    Returns:
        Dictionary with fertilizer information
    """
    db = FertilizerRecommender.FERTILIZER_DB
    
    if fertilizer_name in db:
        info = db[fertilizer_name]
        return {
            'name': fertilizer_name,
            'type': info['type'],
            'npk_ratio': {
                'nitrogen': info['npk'][0],
                'phosphorus': info['npk'][1],
                'potassium': info['npk'][2]
            },
            'application': info['application'],
            'notes': info['notes']
        }
    
    return {
        'name': fertilizer_name,
        'error': 'Fertilizer not found in database'
    }
