"""
Crop Yield Prediction Model for Kalpataru.
Supports both XGBoost and Random Forest models for yield prediction.
"""

from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Global model variables
_model = None
_encoders = None


def get_model_path() -> Path:
    """Get the path to the yield model directory."""
    return Path(__file__).parent


def load_models():
    """Load the trained yield prediction model and encoders."""
    global _model, _encoders
    
    if _model is not None:
        return _model, _encoders
    
    model_dir = get_model_path()
    
    try:
        # Try loading Random Forest model first
        rf_path = model_dir / 'yield_predictor_model.pkl'
        if rf_path.exists():
            _model = joblib.load(rf_path)
            _encoders = {
                'area': joblib.load(model_dir / 'area_encoder.pkl'),
                'item': joblib.load(model_dir / 'item_encoder.pkl')
            }
            logger.info("Yield prediction Random Forest model loaded successfully")
            return _model, _encoders
        
        # Try XGBoost model
        xgb_path = model_dir / 'crop_yield_xgboost.json'
        if xgb_path.exists():
            import xgboost as xgb
            _model = xgb.Booster()
            _model.load_model(str(xgb_path))
            logger.info("Yield prediction XGBoost model loaded successfully")
            return _model, _encoders
            
    except Exception as e:
        logger.error(f"Error loading yield model: {str(e)}")
        
    return None, None


class YieldPredictor:
    """
    Crop yield prediction class that combines ML model predictions
    with rule-based fallback for robust predictions.
    """
    
    # Crop base yields (quintals per hectare)
    BASE_YIELDS = {
        'rice': 45,
        'wheat': 40,
        'maize': 35,
        'cotton': 20,
        'sugarcane': 350,
        'potato': 200,
        'tomato': 150,
        'onion': 180,
        'mustard': 15,
        'soybean': 20,
        'groundnut': 25,
        'barley': 30,
        'millets': 15,
        'paddy': 45
    }
    
    # Region multipliers
    REGION_MULTIPLIERS = {
        'punjab': 1.15,
        'haryana': 1.10,
        'up': 1.0,
        'uttar pradesh': 1.0,
        'maharashtra': 0.95,
        'karnataka': 0.90,
        'tamil nadu': 1.0,
        'tn': 1.0,
        'andhra pradesh': 1.0,
        'ap': 1.0,
        'gujarat': 1.05,
        'madhya pradesh': 0.95,
        'mp': 0.95,
        'rajasthan': 0.85,
        'west bengal': 1.0,
        'kerala': 0.90,
        'odisha': 0.85,
        'assam': 0.80,
        'bihar': 0.90
    }
    
    # Soil type factors
    SOIL_FACTORS = {
        'alluvial': 1.10,
        'black': 1.0,
        'red': 0.90,
        'laterite': 0.85,
        'sandy': 0.75,
        'clay': 0.95,
        'loamy': 1.05
    }
    
    # Season factors
    SEASON_FACTORS = {
        'kharif': 1.0,
        'rabi': 0.95,
        'zaid': 1.05
    }
    
    def __init__(self):
        """Initialize the yield predictor."""
        self.model, self.encoders = load_models()
    
    def predict(self, crop_type: str, area_hectares: float, 
                region: str = None, season: str = None, 
                soil_type: str = None, rainfall: float = None,
                temperature: float = None, pesticide: float = None) -> Dict[str, Any]:
        """
        Predict crop yield.
        
        Args:
            crop_type: Type of crop
            area_hectares: Area in hectares
            region: Geographic region
            season: Growing season (kharif, rabi, zaid)
            soil_type: Type of soil
            rainfall: Annual rainfall in mm
            temperature: Average temperature in Celsius
            pesticide: Pesticide usage in tonnes
            
        Returns:
            Dictionary with yield prediction
        """
        # Try ML model first if available
        if self.model is not None and self.encoders is not None:
            try:
                result = self._predict_ml(
                    crop_type, area_hectares, region, 
                    rainfall, temperature, pesticide
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"ML prediction failed: {str(e)}. Falling back to rule-based.")
        
        # Fallback to rule-based prediction
        return self._predict_rule_based(
            crop_type, area_hectares, region, 
            season, soil_type, rainfall, temperature
        )
    
    def _predict_ml(self, crop_type: str, area_hectares: float,
                    region: str, rainfall: float, 
                    temperature: float, pesticide: float) -> Optional[Dict[str, Any]]:
        """Use ML model for prediction."""
        try:
            # Encode categorical inputs
            area_encoded = 0
            item_encoded = 0
            
            if region and 'area' in self.encoders:
                try:
                    area_encoded = self.encoders['area'].transform([region])[0]
                except ValueError:
                    area_encoded = 0
            
            if crop_type and 'item' in self.encoders:
                try:
                    item_encoded = self.encoders['item'].transform([crop_type])[0]
                except ValueError:
                    item_encoded = 0
            
            # Default values for missing parameters
            rainfall = rainfall or 1000.0
            temperature = temperature or 25.0
            pesticide = pesticide or 50.0
            
            # Create input array
            input_data = np.array([[
                area_encoded, item_encoded, 
                rainfall, pesticide, temperature
            ]])
            
            # Predict
            prediction = self.model.predict(input_data)[0]
            
            # Calculate yield per hectare
            yield_per_ha = prediction
            
            return self._build_result(
                crop_type, area_hectares, yield_per_ha,
                region, None, None, 'ML'
            )
            
        except Exception as e:
            logger.error(f"ML prediction error: {str(e)}")
            return None
    
    def _predict_rule_based(self, crop_type: str, area_hectares: float,
                            region: str, season: str, 
                            soil_type: str, rainfall: float,
                            temperature: float) -> Dict[str, Any]:
        """Rule-based yield prediction."""
        # Get base yield for crop
        crop_lower = crop_type.lower()
        base_yield = self.BASE_YIELDS.get(crop_lower, 30)
        
        # Apply region factor
        region_factor = 1.0
        if region:
            region_lower = region.lower()
            region_factor = self.REGION_MULTIPLIERS.get(region_lower, 1.0)
        
        # Apply soil factor
        soil_factor = 1.0
        if soil_type:
            soil_lower = soil_type.lower()
            soil_factor = self.SOIL_FACTORS.get(soil_lower, 1.0)
        
        # Apply season factor
        season_factor = 1.0
        if season:
            season_lower = season.lower()
            season_factor = self.SEASON_FACTORS.get(season_lower, 1.0)
        
        # Apply rainfall factor
        rainfall_factor = 1.0
        if rainfall:
            if rainfall < 400:
                rainfall_factor = 0.7
            elif rainfall < 600:
                rainfall_factor = 0.85
            elif rainfall < 1000:
                rainfall_factor = 1.0
            elif rainfall < 1500:
                rainfall_factor = 1.1
            else:
                rainfall_factor = 1.0
        
        # Apply temperature factor
        temp_factor = 1.0
        if temperature:
            if temperature < 15:
                temp_factor = 0.8
            elif temperature < 20:
                temp_factor = 0.9
            elif temperature < 30:
                temp_factor = 1.0
            elif temperature < 35:
                temp_factor = 0.95
            else:
                temp_factor = 0.85
        
        # Calculate predicted yield
        yield_per_ha = (
            base_yield * region_factor * soil_factor * 
            season_factor * rainfall_factor * temp_factor
        )
        
        return self._build_result(
            crop_type, area_hectares, yield_per_ha,
            region, soil_type, season, 'Rule-based'
        )
    
    def _build_result(self, crop_type: str, area_hectares: float,
                      yield_per_ha: float, region: str, 
                      soil_type: str, season: str,
                      method: str) -> Dict[str, Any]:
        """Build the result dictionary."""
        total_yield = yield_per_ha * area_hectares
        
        return {
            'crop_type': crop_type,
            'area_hectares': area_hectares,
            'predicted_yield_quintals': round(total_yield, 2),
            'yield_per_hectare': round(yield_per_ha, 2),
            'confidence': 0.85 if method == 'ML' else 0.70,
            'prediction_method': method,
            'input_parameters': {
                'region': region,
                'soil_type': soil_type,
                'season': season
            },
            'factors': {
                'region_factor': self.REGION_MULTIPLIERS.get(region.lower() if region else '', 1.0),
                'soil_factor': self.SOIL_FACTORS.get(soil_type.lower() if soil_type else '', 1.0),
                'season_factor': self.SEASON_FACTORS.get(season.lower() if season else '', 1.0)
            },
            'recommendations': self._get_recommendations(crop_type, yield_per_ha)
        }
    
    def _get_recommendations(self, crop_type: str, predicted_yield: float) -> List[str]:
        """Get recommendations based on predicted yield."""
        base_yield = self.BASE_YIELDS.get(crop_type.lower(), 30)
        
        recommendations = []
        
        if predicted_yield < base_yield * 0.7:
            recommendations.extend([
                'Consider soil testing to identify nutrient deficiencies',
                'Improve irrigation management',
                'Use high-yielding variety seeds',
                'Apply appropriate fertilizers based on crop requirements'
            ])
        elif predicted_yield < base_yield * 0.9:
            recommendations.extend([
                'Optimize fertilizer application timing',
                'Monitor for pest and disease pressure',
                'Consider precision agriculture techniques'
            ])
        else:
            recommendations.extend([
                'Maintain current agricultural practices',
                'Continue regular monitoring',
                'Document successful practices for future reference'
            ])
        
        return recommendations
    
    def get_crop_list(self) -> List[str]:
        """Get list of supported crops."""
        return list(self.BASE_YIELDS.keys())
    
    def get_region_list(self) -> List[str]:
        """Get list of supported regions."""
        return list(self.REGION_MULTIPLIERS.keys())


def predict_yield(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to predict crop yield.
    
    Args:
        data: Dictionary containing input parameters:
            - crop_type: str
            - area_hectares: float
            - region: str (optional)
            - season: str (optional)
            - soil_type: str (optional)
            - rainfall: float (optional)
            - temperature: float (optional)
            - pesticide: float (optional)
            
    Returns:
        Yield prediction dictionary
    """
    predictor = YieldPredictor()
    
    return predictor.predict(
        crop_type=data.get('crop_type', 'rice'),
        area_hectares=data.get('area_hectares', 1.0),
        region=data.get('region'),
        season=data.get('season'),
        soil_type=data.get('soil_type'),
        rainfall=data.get('rainfall'),
        temperature=data.get('temperature'),
        pesticide=data.get('pesticide')
    )


def get_supported_crops() -> List[str]:
    """Get list of supported crops for yield prediction."""
    return list(YieldPredictor.BASE_YIELDS.keys())


def get_supported_regions() -> List[str]:
    """Get list of supported regions for yield prediction."""
    return list(YieldPredictor.REGION_MULTIPLIERS.keys())
