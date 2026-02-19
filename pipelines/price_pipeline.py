"""
Price Data Processing Pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)

def fetch_historical_prices(commodity: str, market: str, days: int = 90) -> pd.DataFrame:
    """
    Fetch historical price data.
    
    Args:
        commodity: Commodity name
        market: Market name
        days: Number of days of historical data
        
    Returns:
        DataFrame with price data
    """
    # In production, fetch from market API
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
             for i in range(days, 0, -1)]
    
    # Generate mock prices with trend
    base_price = 2000
    prices = [base_price + (i % 20 - 10) * 10 for i in range(days)]
    
    data = {
        'date': dates,
        'price': prices,
        'volume': [1000 + (i % 500) for i in range(days)]
    }
    
    return pd.DataFrame(data)

def preprocess_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess price data for model input.
    
    Args:
        df: Raw price DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Create features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Price-based features
    df['price_lag_1'] = df['price'].shift(1)
    df['price_lag_7'] = df['price'].shift(7)
    df['price_rolling_mean_7'] = df['price'].rolling(7).mean()
    df['price_rolling_std_7'] = df['price'].rolling(7).std()
    
    # Fill missing values
    df = df.fillna(method='ffill')
    
    return df

def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Prophet model.
    
    Args:
        df: Price DataFrame
        
    Returns:
        DataFrame with 'ds' and 'y' columns
    """
    prophet_df = df[['date', 'price']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def get_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract seasonality features from price data.
    
    Args:
        df: Price DataFrame
        
    Returns:
        DataFrame with seasonality features
    """
    df = df.copy()
    
    # Monthly seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Weekly seasonality
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

import numpy as np
