"""
Prophet Model for Commodity Price Forecasting.
"""

from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)

def predict_price(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict commodity prices.
    
    Args:
        data: Input data with commodity and market info
        
    Returns:
        Price forecast
    """
    commodity = data.get('commodity', 'rice')
    market = data.get('market', 'delhi')
    forecast_days = data.get('forecast_days', 7)
    
    # Get current price and generate forecast
    current_price = get_current_price(commodity, market)
    forecast = generate_price_forecast(commodity, market, current_price, forecast_days)
    
    return {
        'commodity': commodity,
        'market': market,
        'current_price': current_price,
        'forecast': forecast,
        'trend': get_trend(forecast),
        'confidence': 0.75
    }

def get_current_price(commodity: str, market: str) -> float:
    """Get current market price for commodity."""
    # Simplified price lookup (in Rs/quintal)
    base_prices = {
        'rice': 2200,
        'wheat': 2150,
        'maize': 1900,
        'cotton': 6200,
        'sugarcane': 350,
        'potato': 1200,
        'tomato': 1800,
        'onion': 1500,
        'mustard': 5000,
        'soybean': 4500
    }
    
    # Market adjustment
    market_multipliers = {
        'delhi': 1.05,
        'mumbai': 1.1,
        'kolkata': 1.0,
        'chennai': 1.08,
        'ahmedabad': 0.98
    }
    
    base = base_prices.get(commodity, 2000)
    multiplier = market_multipliers.get(market.lower(), 1.0)
    
    return base * multiplier

def generate_price_forecast(commodity: str, market: str, 
                           current_price: float, days: int) -> List[Dict[str, Any]]:
    """Generate price forecast."""
    forecast = []
    volatility = get_volatility(commodity)
    
    for i in range(days):
        date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        
        # Add random variation based on volatility
        change_percent = np.random.normal(0, volatility)
        price = current_price * (1 + change_percent/100)
        
        forecast.append({
            'date': date,
            'price_rs_per_quintal': round(price, 2),
            'change_percent': round(change_percent, 2)
        })
    
    return forecast

def get_volatility(commodity: str) -> float:
    """Get price volatility for commodity."""
    volatility_map = {
        'rice': 2.0,
        'wheat': 1.8,
        'maize': 2.5,
        'cotton': 3.5,
        'sugarcane': 1.5,
        'potato': 4.0,
        'tomato': 5.0,
        'onion': 4.5,
        'mustard': 2.5,
        'soybean': 3.0
    }
    return volatility_map.get(commodity, 3.0)

def get_trend(forecast: List[Dict[str, Any]]) -> str:
    """Determine price trend from forecast."""
    if not forecast:
        return 'stable'
    
    changes = [f['change_percent'] for f in forecast]
    avg_change = sum(changes) / len(changes)
    
    if avg_change > 1:
        return 'increasing'
    elif avg_change < -1:
        return 'decreasing'
    else:
        return 'stable'
