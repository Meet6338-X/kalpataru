"""
Yield Prediction Data Processing Pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

def fetch_yield_data(crop: str, region: str, season: str) -> pd.DataFrame:
    """
    Fetch historical yield data.
    
    Args:
        crop: Crop type
        region: Region
        season: Season
        
    Returns:
        DataFrame with yield data
    """
    # In production, fetch from agricultural database
    # Mock data for now
    years = list(range(2015, 2025))
    
    data = {
        'year': years,
        'yield_quintals_per_hectare': [40 + (i % 5) for i in range(len(years))],
        'area_hectares': [1000 + (i * 100) for i in range(len(years))],
        'rainfall_mm': [1000 + (i % 200 - 100) for i in range(len(years))],
        'temperature_avg': [25 + (i % 3 - 1) for i in range(len(years))]
    }
    
    return pd.DataFrame(data)

def preprocess_yield_features(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Preprocess input features for yield prediction.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Feature dictionary
    """
    # Encode categorical variables
    soil_encoding = {
        'alluvial': 1, 'black': 2, 'red': 3, 
        'laterite': 4, 'sandy': 5, 'clay': 6
    }
    
    season_encoding = {
        'kharif': 1, 'rabi': 2, 'zaid': 3
    }
    
    region_encoding = {
        'punjab': 1, 'haryana': 2, 'up': 3,
        'maharashtra': 4, 'karnataka': 5, 'tn': 6
    }
    
    crop_encoding = {
        'rice': 1, 'wheat': 2, 'maize': 3,
        'cotton': 4, 'sugarcane': 5, 'potato': 6
    }
    
    features = {
        'area_hectares': data.get('area_hectares', 1.0),
        'soil_type': soil_encoding.get(data.get('soil_type', 'alluvial'), 1),
        'season': season_encoding.get(data.get('season', 'kharif'), 1),
        'region': region_encoding.get(data.get('region', 'punjab'), 1),
        'crop_type': crop_encoding.get(data.get('crop_type', 'rice'), 1)
    }
    
    return features

def create_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for yield prediction.
    
    Args:
        df: Historical yield DataFrame
        
    Returns:
        DataFrame with features
    """
    df = df.copy()
    
    # Create lag features
    df['yield_lag_1'] = df['yield_quintals_per_hectare'].shift(1)
    df['yield_lag_2'] = df['yield_quintals_per_hectare'].shift(2)
    
    # Rolling statistics
    df['yield_rolling_mean_3'] = df['yield_quintals_per_hectare'].rolling(3).mean()
    df['yield_rolling_std_3'] = df['yield_quintals_per_hectare'].rolling(3).std()
    
    # Weather interaction
    df['rainfall_temp'] = df['rainfall_mm'] * df['temperature_avg']
    
    # Fill missing values
    df = df.fillna(method='ffill')
    
    return df

def get_yield_factors(yield_prediction: float, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze factors affecting yield prediction.
    
    Args:
        yield_prediction: Predicted yield
        data: Input features
        
    Returns:
        Factor analysis
    """
    factors = []
    
    # Soil factor
    soil = data.get('soil_type', 'alluvial')
    if soil == 'alluvial':
        factors.append({
            'factor': 'Soil Type',
            'impact': 'positive',
            'description': 'Alluvial soil is highly fertile'
        })
    
    # Season factor
    season = data.get('season', 'kharif')
    factors.append({
        'factor': 'Season',
        'impact': 'neutral',
        'description': f'{season.capitalize()} season conditions expected'
    })
    
    return factors
