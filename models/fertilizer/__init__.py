"""
Fertilizer Recommendation Module for Kalpataru.
Provides ML-based fertilizer recommendations based on soil and crop parameters.
"""

from models.fertilizer.fertilizer_model import (
    predict_fertilizer,
    get_fertilizer_info,
    FertilizerRecommender
)

__all__ = [
    'predict_fertilizer',
    'get_fertilizer_info',
    'FertilizerRecommender'
]
