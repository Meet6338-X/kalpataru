"""
Soil Classification Model for Kalpataru.
CNN-based soil type classification from soil images.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Global model variable
_model = None


def get_model_path() -> Path:
    """Get the path to the soil model directory."""
    return Path(__file__).parent


class SoilClassifier:
    """
    Soil classification class that uses CNN for image-based soil type prediction.
    Includes fallback to color-based analysis when model is not available.
    """
    
    # Soil types supported
    SOIL_TYPES = [
        'Alluvial',
        'Black', 
        'Clay',
        'Laterite',
        'Red',
        'Sandy'
    ]
    
    # Soil characteristics database
    SOIL_DATABASE = {
        'Alluvial': {
            'color_range': ('#8B7355', '#D2B48C'),  # Light brown to tan
            'characteristics': [
                'Highly fertile',
                'Good water retention',
                'Suitable for most crops',
                'Rich in potash'
            ],
            'recommended_crops': ['Wheat', 'Rice', 'Sugarcane', 'Cotton', 'Vegetables'],
            'ph_range': (6.5, 8.4),
            'texture': 'Fine to medium',
            'drainage': 'Good'
        },
        'Black': {
            'color_range': ('#1C1C1C', '#4A4A4A'),  # Dark black to dark gray
            'characteristics': [
                'High clay content',
                'Self-plowing when wet/dry',
                'Rich in lime, iron, magnesia',
                'Poor drainage'
            ],
            'recommended_crops': ['Cotton', 'Sugarcane', 'Wheat', 'Millets'],
            'ph_range': (7.0, 8.5),
            'texture': 'Fine',
            'drainage': 'Poor to moderate'
        },
        'Clay': {
            'color_range': ('#8B4513', '#A0522D'),  # Brown shades
            'characteristics': [
                'Heavy soil',
                'High water retention',
                'Slow drainage',
                'Nutrient rich'
            ],
            'recommended_crops': ['Rice', 'Broccoli', 'Cabbage', 'Beans'],
            'ph_range': (5.5, 7.5),
            'texture': 'Fine',
            'drainage': 'Poor'
        },
        'Laterite': {
            'color_range': ('#B22222', '#CD5C5C'),  # Reddish shades
            'characteristics': [
                'Rich in iron and aluminum',
                'Low fertility',
                'Acidic nature',
                'Good for plantation crops'
            ],
            'recommended_crops': ['Tea', 'Coffee', 'Rubber', 'Cashew'],
            'ph_range': (4.5, 6.5),
            'texture': 'Coarse',
            'drainage': 'Good'
        },
        'Red': {
            'color_range': ('#CD5C5C', '#F08080'),  # Light red to coral
            'characteristics': [
                'Rich in iron oxide',
                'Porous and friable',
                'Low in nitrogen, phosphorus',
                'Good aeration'
            ],
            'recommended_crops': ['Rice', 'Ragi', 'Groundnut', 'Potato'],
            'ph_range': (5.5, 7.0),
            'texture': 'Medium',
            'drainage': 'Good'
        },
        'Sandy': {
            'color_range': ('#F4A460', '#FAEBD7'),  # Sandy brown to antique white
            'characteristics': [
                'Low nutrient content',
                'Excellent drainage',
                'Warms up quickly',
                'Low water retention'
            ],
            'recommended_crops': ['Groundnut', 'Watermelon', 'Millet', 'Carrots'],
            'ph_range': (6.0, 7.5),
            'texture': 'Coarse',
            'drainage': 'Excellent'
        }
    }
    
    def __init__(self):
        """Initialize the soil classifier."""
        self.model = self._load_model()
        self.image_size = (224, 224)
    
    def _load_model(self):
        """Load the trained CNN model for soil classification."""
        global _model
        
        if _model is not None:
            return _model
        
        model_path = get_model_path() / 'soil_classifier_model.h5'
        
        try:
            if model_path.exists():
                from tensorflow.keras.models import load_model
                _model = load_model(str(model_path))
                logger.info("Soil classification model loaded successfully")
                return _model
            else:
                logger.warning("Soil classifier model not found. Using color-based analysis.")
                return None
        except Exception as e:
            logger.warning(f"Could not load soil model: {str(e)}. Using color-based analysis.")
            return None
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image for model prediction.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image array
        """
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize(self.image_size)
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Classify soil type from image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with soil classification results
        """
        if self.model is not None:
            return self._predict_cnn(image_data)
        else:
            return self._predict_color_based(image_data)
    
    def _predict_cnn(self, image_data: bytes) -> Dict[str, Any]:
        """Use CNN model for prediction."""
        try:
            processed_image = self.preprocess_image(image_data)
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            soil_type = self.SOIL_TYPES[predicted_class]
            
            return self._build_result(soil_type, confidence, predictions[0])
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {str(e)}")
            return self._predict_color_based(image_data)
    
    def _predict_color_based(self, image_data: bytes) -> Dict[str, Any]:
        """Use color analysis for soil classification (fallback method)."""
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image = image.resize((100, 100))  # Smaller size for faster processing
            
            # Get average color
            pixels = np.array(image)
            avg_color = pixels.mean(axis=(0, 1))
            
            # Analyze color characteristics
            r, g, b = avg_color
            
            # Determine soil type based on color
            soil_type, confidence = self._classify_by_color(r, g, b)
            
            # Generate confidence scores for all classes
            all_scores = self._generate_color_scores(r, g, b)
            
            return self._build_result(soil_type, confidence, all_scores)
            
        except Exception as e:
            logger.error(f"Color-based prediction failed: {str(e)}")
            return self._build_result('Alluvial', 0.5, None)
    
    def _classify_by_color(self, r: float, g: float, b: float) -> Tuple[str, float]:
        """
        Classify soil type based on RGB color values.
        
        Args:
            r: Red channel value
            g: Green channel value
            b: Blue channel value
            
        Returns:
            Tuple of (soil_type, confidence)
        """
        # Calculate color ratios and characteristics
        brightness = (r + g + b) / 3
        red_ratio = r / (g + b + 1)
        warmth = r / (b + 1)
        
        # Classification rules based on typical soil colors
        if brightness < 80 and r < 100 and g < 100:
            # Dark soil - likely Black soil
            return 'Black', 0.75
        
        elif r > 150 and g > 100 and b < 100:
            # Reddish soil
            if brightness > 150:
                return 'Laterite', 0.70
            else:
                return 'Red', 0.72
        
        elif brightness > 180 and r > 180 and g > 160:
            # Light sandy soil
            return 'Sandy', 0.68
        
        elif 100 < brightness < 160 and 80 < r < 150:
            # Medium brown - Alluvial or Clay
            if g > r * 0.85:
                return 'Clay', 0.65
            else:
                return 'Alluvial', 0.70
        
        elif r > g and r > b and r < 150:
            # Reddish-brown
            return 'Red', 0.65
        
        else:
            # Default to Alluvial (most common)
            return 'Alluvial', 0.55
    
    def _generate_color_scores(self, r: float, g: float, b: float) -> np.ndarray:
        """Generate probability-like scores for all soil types."""
        brightness = (r + g + b) / 3
        
        scores = np.zeros(len(self.SOIL_TYPES))
        
        # Score based on color characteristics
        scores[0] = 0.3 if 100 < brightness < 160 else 0.1  # Alluvial
        scores[1] = 0.4 if brightness < 80 else 0.1  # Black
        scores[2] = 0.3 if 80 < brightness < 140 else 0.1  # Clay
        scores[3] = 0.35 if r > 150 and brightness > 140 else 0.1  # Laterite
        scores[4] = 0.35 if r > g and r > b else 0.1  # Red
        scores[5] = 0.35 if brightness > 180 else 0.1  # Sandy
        
        # Normalize to sum to 1
        scores = scores / scores.sum()
        
        return scores
    
    def _build_result(self, soil_type: str, confidence: float, 
                      all_scores: Optional[np.ndarray]) -> Dict[str, Any]:
        """Build the result dictionary with soil information."""
        soil_info = self.SOIL_DATABASE.get(soil_type, {})
        
        result = {
            'soil_type': soil_type,
            'confidence': round(confidence, 4),
            'characteristics': soil_info.get('characteristics', []),
            'recommended_crops': soil_info.get('recommended_crops', []),
            'ph_range': soil_info.get('ph_range', (6.0, 7.5)),
            'texture': soil_info.get('texture', 'Unknown'),
            'drainage': soil_info.get('drainage', 'Unknown'),
            'method': 'CNN' if self.model else 'Color Analysis'
        }
        
        # Add all class probabilities if available
        if all_scores is not None:
            result['all_probabilities'] = {
                soil: round(float(score), 4) 
                for soil, score in zip(self.SOIL_TYPES, all_scores)
            }
        
        return result


def classify_soil(image_file) -> Dict[str, Any]:
    """
    Main function to classify soil from image.
    
    Args:
        image_file: Image file from request
        
    Returns:
        Soil classification result dictionary
    """
    classifier = SoilClassifier()
    
    # Read image data
    image_data = image_file.read()
    
    return classifier.predict(image_data)


def get_soil_info(soil_type: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific soil type.
    
    Args:
        soil_type: Type of soil
        
    Returns:
        Dictionary with soil information
    """
    db = SoilClassifier.SOIL_DATABASE
    
    if soil_type in db:
        info = db[soil_type]
        return {
            'soil_type': soil_type,
            'characteristics': info['characteristics'],
            'recommended_crops': info['recommended_crops'],
            'ph_range': info['ph_range'],
            'texture': info['texture'],
            'drainage': info['drainage']
        }
    
    return {
        'soil_type': soil_type,
        'error': 'Soil type not found in database'
    }
