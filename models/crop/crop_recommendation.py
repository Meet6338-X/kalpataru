"""
Crop Recommendation Model for Kalpataru.
ML-based crop recommendations based on soil and climate parameters.
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
_label_encoder = None
_scaler = None


def get_model_path() -> Path:
    """Get the path to the crop model directory."""
    return Path(__file__).parent


def load_models():
    """Load the trained crop recommendation model and encoders."""
    global _model, _label_encoder, _scaler
    
    if _model is not None:
        return _model, _label_encoder, _scaler
    
    model_dir = get_model_path()
    
    try:
        model_path = model_dir / 'rf_model.pkl'
        encoder_path = model_dir / 'label_encoder.pkl'
        scaler_path = model_dir / 'scaler.pkl'
        
        if model_path.exists():
            _model = joblib.load(model_path)
            _label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None
            _scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            logger.info("Crop recommendation model loaded successfully")
            return _model, _label_encoder, _scaler
            
    except Exception as e:
        logger.error(f"Error loading crop model: {str(e)}")
        
    return None, None, None


class CropRecommender:
    """
    Crop recommendation class that combines ML model predictions
    with rule-based fallback for robust recommendations.
    """
    
    # Crop database with optimal growing conditions
    CROP_DATABASE = {
        'rice': {
            'optimal_ph': [5.5, 7.0],
            'optimal_temp': [20, 35],
            'min_rainfall': 1000,
            'max_rainfall': 3000,
            'soil_types': ['alluvial', 'clay', 'loamy'],
            'seasons': ['kharif'],
            'water_need': 'high',
            'growing_days': [100, 150]
        },
        'wheat': {
            'optimal_ph': [6.0, 7.5],
            'optimal_temp': [15, 25],
            'min_rainfall': 500,
            'max_rainfall': 1000,
            'soil_types': ['alluvial', 'loamy', 'clay'],
            'seasons': ['rabi'],
            'water_need': 'medium',
            'growing_days': [110, 140]
        },
        'maize': {
            'optimal_ph': [5.8, 7.0],
            'optimal_temp': [20, 30],
            'min_rainfall': 600,
            'max_rainfall': 1200,
            'soil_types': ['alluvial', 'loamy', 'red'],
            'seasons': ['kharif', 'rabi'],
            'water_need': 'medium',
            'growing_days': [80, 110]
        },
        'cotton': {
            'optimal_ph': [5.5, 8.0],
            'optimal_temp': [25, 35],
            'min_rainfall': 500,
            'max_rainfall': 1000,
            'soil_types': ['black', 'alluvial', 'red'],
            'seasons': ['kharif'],
            'water_need': 'medium',
            'growing_days': [150, 200]
        },
        'sugarcane': {
            'optimal_ph': [6.0, 8.0],
            'optimal_temp': [24, 32],
            'min_rainfall': 1500,
            'max_rainfall': 2500,
            'soil_types': ['alluvial', 'black', 'loamy'],
            'seasons': ['kharif', 'rabi'],
            'water_need': 'high',
            'growing_days': [300, 400]
        },
        'potato': {
            'optimal_ph': [5.0, 6.5],
            'optimal_temp': [15, 25],
            'min_rainfall': 400,
            'max_rainfall': 800,
            'soil_types': ['alluvial', 'loamy', 'sandy'],
            'seasons': ['rabi'],
            'water_need': 'medium',
            'growing_days': [90, 120]
        },
        'tomato': {
            'optimal_ph': [6.0, 6.8],
            'optimal_temp': [20, 28],
            'min_rainfall': 400,
            'max_rainfall': 800,
            'soil_types': ['loamy', 'alluvial', 'sandy'],
            'seasons': ['kharif', 'rabi', 'zaid'],
            'water_need': 'medium',
            'growing_days': [60, 90]
        },
        'onion': {
            'optimal_ph': [6.0, 7.0],
            'optimal_temp': [20, 30],
            'min_rainfall': 350,
            'max_rainfall': 700,
            'soil_types': ['sandy', 'loamy', 'alluvial'],
            'seasons': ['rabi'],
            'water_need': 'low',
            'growing_days': [90, 120]
        },
        'mustard': {
            'optimal_ph': [6.0, 7.5],
            'optimal_temp': [15, 25],
            'min_rainfall': 400,
            'max_rainfall': 800,
            'soil_types': ['alluvial', 'loamy', 'clay'],
            'seasons': ['rabi'],
            'water_need': 'low',
            'growing_days': [100, 140]
        },
        'soybean': {
            'optimal_ph': [5.5, 7.0],
            'optimal_temp': [20, 30],
            'min_rainfall': 450,
            'max_rainfall': 900,
            'soil_types': ['alluvial', 'black', 'loamy'],
            'seasons': ['kharif'],
            'water_need': 'medium',
            'growing_days': [90, 120]
        },
        'groundnut': {
            'optimal_ph': [5.5, 7.0],
            'optimal_temp': [25, 30],
            'min_rainfall': 500,
            'max_rainfall': 1000,
            'soil_types': ['sandy', 'red', 'alluvial'],
            'seasons': ['kharif'],
            'water_need': 'low',
            'growing_days': [100, 130]
        },
        'barley': {
            'optimal_ph': [6.0, 7.5],
            'optimal_temp': [12, 20],
            'min_rainfall': 400,
            'max_rainfall': 700,
            'soil_types': ['alluvial', 'loamy', 'sandy'],
            'seasons': ['rabi'],
            'water_need': 'low',
            'growing_days': [100, 130]
        },
        'millets': {
            'optimal_ph': [5.5, 7.0],
            'optimal_temp': [25, 35],
            'min_rainfall': 300,
            'max_rainfall': 700,
            'soil_types': ['sandy', 'red', 'loamy'],
            'seasons': ['kharif'],
            'water_need': 'low',
            'growing_days': [70, 100]
        },
        'chickpea': {
            'optimal_ph': [6.0, 8.0],
            'optimal_temp': [15, 25],
            'min_rainfall': 300,
            'max_rainfall': 600,
            'soil_types': ['alluvial', 'black', 'loamy'],
            'seasons': ['rabi'],
            'water_need': 'low',
            'growing_days': [90, 120]
        }
    }
    
    # Soil type suitability scores
    SOIL_SUITABILITY = {
        'alluvial': {'rice': 10, 'wheat': 10, 'sugarcane': 9, 'cotton': 7, 'maize': 8},
        'black': {'cotton': 10, 'sugarcane': 8, 'wheat': 6, 'maize': 7, 'soybean': 9},
        'red': {'maize': 8, 'cotton': 7, 'groundnut': 8, 'millets': 9},
        'laterite': {'tea': 10, 'coffee': 8, 'rubber': 7, 'cashew': 9},
        'sandy': {'groundnut': 9, 'watermelon': 8, 'millet': 7, 'onion': 8},
        'clay': {'rice': 9, 'wheat': 7, 'sugarcane': 8},
        'loamy': {'wheat': 9, 'maize': 9, 'vegetables': 10, 'potato': 9}
    }
    
    def __init__(self):
        """Initialize the crop recommender."""
        self.model, self.label_encoder, self.scaler = load_models()
    
    def recommend(self, n: float, p: float, k: float, 
                  temperature: float, humidity: float, 
                  ph: float, rainfall: float,
                  soil_type: str = None, season: str = None) -> Dict[str, Any]:
        """
        Recommend best crops for given conditions.
        
        Args:
            n: Nitrogen level in soil
            p: Phosphorus level in soil
            k: Potassium level in soil
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            ph: Soil pH level
            rainfall: Rainfall in mm
            soil_type: Type of soil (optional)
            season: Growing season (optional)
            
        Returns:
            Dictionary with crop recommendations
        """
        # Try ML model first if available
        if self.model is not None:
            try:
                result = self._predict_ml(n, p, k, temperature, humidity, ph, rainfall)
                if result:
                    # Enhance with rule-based scoring
                    result = self._enhance_with_rules(result, soil_type, season)
                    return result
            except Exception as e:
                logger.warning(f"ML prediction failed: {str(e)}. Falling back to rule-based.")
        
        # Fallback to rule-based prediction
        return self._predict_rule_based(
            n, p, k, temperature, humidity, ph, rainfall, soil_type, season
        )
    
    def _predict_ml(self, n: float, p: float, k: float,
                    temperature: float, humidity: float,
                    ph: float, rainfall: float) -> Optional[Dict[str, Any]]:
        """Use ML model for prediction."""
        try:
            # Prepare input
            input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            
            # Scale if scaler available
            if self.scaler is not None:
                input_data = self.scaler.transform(input_data)
            
            # Predict
            prediction = self.model.predict(input_data)[0]
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_data)[0]
            
            # Decode label
            if self.label_encoder is not None:
                crop = self.label_encoder.inverse_transform([prediction])[0]
            else:
                crop = prediction
            
            # Build result
            result = {
                'best_crop': crop,
                'confidence': float(max(probabilities)) if probabilities is not None else 0.85,
                'method': 'ML'
            }
            
            # Add all probabilities
            if probabilities is not None and self.label_encoder is not None:
                crops = self.label_encoder.classes_
                result['all_probabilities'] = {
                    crops[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"ML prediction error: {str(e)}")
            return None
    
    def _predict_rule_based(self, n: float, p: float, k: float,
                            temperature: float, humidity: float,
                            ph: float, rainfall: float,
                            soil_type: str, season: str) -> Dict[str, Any]:
        """Rule-based crop recommendation."""
        crops = list(self.CROP_DATABASE.keys())
        scored_crops = []
        
        for crop in crops:
            crop_info = self.CROP_DATABASE[crop]
            score = self._calculate_crop_score(
                crop, crop_info, n, p, k, 
                temperature, humidity, ph, rainfall,
                soil_type, season
            )
            
            scored_crops.append({
                'crop': crop,
                'score': round(score, 2),
                'suitability': self._get_suitability_label(score),
                'reasons': self._get_reasons(crop, crop_info, ph, rainfall, temperature, soil_type)
            })
        
        # Sort by score
        scored_crops.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'recommendations': scored_crops[:5],
            'best_crop': scored_crops[0]['crop'] if scored_crops else None,
            'method': 'Rule-based',
            'input_conditions': {
                'N': n, 'P': p, 'K': k,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall,
                'soil_type': soil_type,
                'season': season
            }
        }
    
    def _enhance_with_rules(self, ml_result: Dict[str, Any], 
                            soil_type: str, season: str) -> Dict[str, Any]:
        """Enhance ML result with rule-based adjustments."""
        if 'all_probabilities' in ml_result:
            # Adjust probabilities based on soil and season
            adjusted = {}
            for crop, prob in ml_result['all_probabilities'].items():
                adjustment = 1.0
                
                # Soil adjustment
                if soil_type and soil_type.lower() in self.SOIL_SUITABILITY:
                    soil_score = self.SOIL_SUITABILITY[soil_type.lower()].get(crop, 5)
                    adjustment *= (soil_score / 10)
                
                # Season adjustment
                if season:
                    crop_info = self.CROP_DATABASE.get(crop, {})
                    if season.lower() in crop_info.get('seasons', []):
                        adjustment *= 1.1
                    else:
                        adjustment *= 0.8
                
                adjusted[crop] = prob * adjustment
            
            # Re-sort
            sorted_crops = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
            
            ml_result['adjusted_probabilities'] = adjusted
            ml_result['recommendations'] = [
                {
                    'crop': crop,
                    'score': round(score * 100, 2),
                    'suitability': self._get_suitability_label(score * 100)
                }
                for crop, score in sorted_crops[:5]
            ]
            ml_result['best_crop'] = sorted_crops[0][0] if sorted_crops else ml_result['best_crop']
        
        return ml_result
    
    def _calculate_crop_score(self, crop: str, crop_info: Dict,
                              n: float, p: float, k: float,
                              temperature: float, humidity: float,
                              ph: float, rainfall: float,
                              soil_type: str, season: str) -> float:
        """Calculate suitability score for a crop."""
        score = 100.0
        
        # pH scoring
        ph_min, ph_max = crop_info['optimal_ph']
        if ph < ph_min:
            score -= (ph_min - ph) * 10
        elif ph > ph_max:
            score -= (ph - ph_max) * 10
        
        # Temperature scoring
        temp_min, temp_max = crop_info['optimal_temp']
        if temperature < temp_min:
            score -= (temp_min - temperature) * 3
        elif temperature > temp_max:
            score -= (temperature - temp_max) * 3
        
        # Rainfall scoring
        min_rain = crop_info['min_rainfall']
        max_rain = crop_info['max_rainfall']
        if rainfall < min_rain:
            score -= (min_rain - rainfall) / 50
        elif rainfall > max_rain:
            score -= (rainfall - max_rain) / 100
        
        # Soil type scoring
        if soil_type:
            soil_score = self.SOIL_SUITABILITY.get(soil_type.lower(), {}).get(crop, 5)
            score += (soil_score - 5) * 2
        
        # Season scoring
        if season:
            if season.lower() in crop_info.get('seasons', []):
                score += 10
            else:
                score -= 20
        
        # Nutrient scoring
        # Higher N is good for leafy crops, P for root development, K for fruit
        if crop in ['rice', 'wheat', 'maize']:
            score += min(n / 10, 5)  # N bonus for cereals
        if crop in ['potato', 'onion', 'groundnut']:
            score += min(p / 10, 5)  # P bonus for root crops
        if crop in ['tomato', 'cotton', 'sugarcane']:
            score += min(k / 10, 5)  # K bonus for fruit/fiber crops
        
        return max(0, min(100, score))
    
    def _get_suitability_label(self, score: float) -> str:
        """Get suitability label from score."""
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 50:
            return 'moderate'
        elif score >= 25:
            return 'poor'
        else:
            return 'unsuitable'
    
    def _get_reasons(self, crop: str, crop_info: Dict,
                     ph: float, rainfall: float, 
                     temperature: float, soil_type: str) -> List[str]:
        """Get reasons for crop recommendation."""
        reasons = []
        
        # pH reasons
        ph_min, ph_max = crop_info['optimal_ph']
        if ph_min <= ph <= ph_max:
            reasons.append(f"Soil pH ({ph}) is optimal for {crop}")
        else:
            reasons.append(f"pH adjustment may be needed for optimal {crop} growth")
        
        # Rainfall reasons
        if rainfall >= crop_info['min_rainfall']:
            reasons.append("Adequate rainfall for this crop")
        else:
            reasons.append("Irrigation may be required")
        
        # Temperature reasons
        temp_min, temp_max = crop_info['optimal_temp']
        if temp_min <= temperature <= temp_max:
            reasons.append("Temperature is suitable")
        else:
            reasons.append("Temperature is outside optimal range")
        
        # Soil reasons
        if soil_type and soil_type.lower() in crop_info.get('soil_types', []):
            reasons.append(f"{soil_type.title()} soil is suitable for {crop}")
        
        return reasons
    
    def get_crop_info(self, crop: str) -> Dict[str, Any]:
        """Get detailed information about a specific crop."""
        return self.CROP_DATABASE.get(crop.lower(), {})
    
    def get_all_crops(self) -> List[str]:
        """Get list of all supported crops."""
        return list(self.CROP_DATABASE.keys())


def recommend_crop(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to recommend crops.
    
    Args:
        data: Dictionary containing input parameters:
            - N: Nitrogen level
            - P: Phosphorus level
            - K: Potassium level
            - temperature: Temperature in Celsius
            - humidity: Humidity percentage
            - ph: Soil pH
            - rainfall: Rainfall in mm
            - soil_type: Type of soil (optional)
            - season: Growing season (optional)
            
    Returns:
        Crop recommendation dictionary
    """
    recommender = CropRecommender()
    
    return recommender.recommend(
        n=data.get('N', data.get('n', 50)),
        p=data.get('P', data.get('p', 50)),
        k=data.get('K', data.get('k', 50)),
        temperature=data.get('temperature', 25),
        humidity=data.get('humidity', 60),
        ph=data.get('ph', 7.0),
        rainfall=data.get('rainfall', 1000),
        soil_type=data.get('soil_type'),
        season=data.get('season')
    )


def get_supported_crops() -> List[str]:
    """Get list of supported crops for recommendation."""
    return list(CropRecommender.CROP_DATABASE.keys())


def get_crop_requirements(crop: str) -> Dict[str, Any]:
    """Get growing requirements for a specific crop."""
    return CropRecommender.CROP_DATABASE.get(crop.lower(), {})
