"""
Soil Classification Module for Kalpataru.
Provides CNN-based soil type classification from images.
"""

from models.soil.soil_classifier import (
    classify_soil,
    get_soil_info,
    SoilClassifier
)

__all__ = [
    'classify_soil',
    'get_soil_info',
    'SoilClassifier'
]
