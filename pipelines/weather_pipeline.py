"""
Weather Data Processing Pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)

def fetch_historical_weather(location: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch historical weather data.
    
    Args:
        location: Location string
        days: Number of days of historical data
        
    Returns:
        DataFrame with weather data
    """
    # In production, this would fetch from weather API
    # For now, generate mock data
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
             for i in range(days, 0, -1)]
    
    data = {
        'date': dates,
        'temperature': [25 + (i % 10 - 5) for i in range(days)],
        'humidity': [60 + (i % 20 - 10) for i in range(days)],
        'rainfall': [0 if i % 5 != 0 else 5 + i % 10 for i in range(days)],
        'wind_speed': [10 + (i % 10) for i in range(days)]
    }
    
    return pd.DataFrame(data)

def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess weather data for model input.
    
    Args:
        df: Raw weather DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # Add rolling averages
    df['temp_rolling_7d'] = df['temperature'].rolling(7).mean()
    df['humidity_rolling_7d'] = df['humidity'].rolling(7).mean()
    
    # Fill missing values
    df = df.fillna(method='ffill')
    
    return df

def create_sequences(data: pd.DataFrame, seq_length: int = 7) -> tuple:
    """
    Create sequences for LSTM model.
    
    Args:
        data: Preprocessed weather DataFrame
        seq_length: Length of sequence
        
    Returns:
        X and y arrays
    """
    features = ['temperature', 'humidity', 'rainfall', 'wind_speed']
    
    X, y = [], []
    values = data[features].values
    
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length])
    
    return np.array(X), np.array(y)

def get_weather_features(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract weather features from API response.
    
    Args:
        data: Weather API response
        
    Returns:
        Feature dictionary
    """
    return {
        'temperature': data.get('temperature', 25),
        'humidity': data.get('humidity', 60),
        'rainfall': data.get('rainfall', 0),
        'wind_speed': data.get('wind_speed', 10)
    }
