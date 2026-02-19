"""
LSTM Model for Weather Forecasting.
"""

from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timedelta

from utils.logger import setup_logger

logger = setup_logger(__name__)

def predict_weather(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict weather for upcoming days.
    
    Args:
        data: Input data with location and historical info
        
    Returns:
        Weather forecast
    """
    location = data.get('location', 'unknown')
    forecast_days = data.get('forecast_days', 7)
    
    # Generate mock forecast for demonstration
    # In production, this would use a trained LSTM model
    forecast = generate_mock_forecast(forecast_days)
    
    return {
        'location': location,
        'forecast': forecast,
        'summary': get_summary(forecast),
        'confidence': 0.78
    }

def generate_mock_forecast(days: int) -> List[Dict[str, Any]]:
    """Generate mock weather forecast."""
    forecast = []
    base_temp = 28
    base_humidity = 65
    
    for i in range(days):
        date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        
        # Add some variation
        temp_variation = np.random.randint(-3, 4)
        humidity_variation = np.random.randint(-10, 11)
        
        forecast.append({
            'date': date,
            'temperature_celsius': base_temp + temp_variation,
            'humidity_percent': max(30, min(95, base_humidity + humidity_variation)),
            'condition': np.random.choice(['sunny', 'cloudy', 'partly_cloudy', 'rainy']),
            'precipitation_mm': max(0, humidity_variation * 0.5),
            'wind_speed_kmh': np.random.randint(5, 20)
        })
    
    return forecast

def get_summary(forecast: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary from forecast."""
    temps = [f['temperature_celsius'] for f in forecast]
    humidity = [f['humidity_percent'] for f in forecast]
    
    return {
        'avg_temperature': round(sum(temps) / len(temps), 1),
        'min_temperature': min(temps),
        'max_temperature': max(temps),
        'avg_humidity': round(sum(humidity) / len(humidity), 1),
        'rainy_days': sum(1 for f in forecast if f['condition'] == 'rainy')
    }
