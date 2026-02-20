"""
Configuration settings for Kalpataru application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
SCRIPTS_DIR = BASE_DIR / 'scripts'

# API settings
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 5000))
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# =============================================================================
# Data paths
# =============================================================================
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'
ANALYSIS_DIR = DATA_DIR / 'analysis'

# Raw data subdirectories
RAW_CROP_RECOMMENDATION_DIR = RAW_DATA_DIR / 'crop_recommendation'
RAW_CROP_YIELD_DIR = RAW_DATA_DIR / 'crop_yield'
RAW_DISEASE_IMAGES_DIR = RAW_DATA_DIR / 'disease_images'

# Processed data subdirectories
PROCESSED_CROP_RECOMMENDATION_DIR = PROCESSED_DATA_DIR / 'crop_recommendation'
PROCESSED_CROP_YIELD_DIR = PROCESSED_DATA_DIR / 'crop_yield'
PROCESSED_DISEASE_IMAGES_DIR = PROCESSED_DATA_DIR / 'disease_images'

# =============================================================================
# Model settings
# =============================================================================
# Disease Detection
DISEASE_MODEL_PATH = MODEL_DIR / 'disease' / 'disease_mobilenet_model.h5'
DISEASE_WEIGHTS_PATH = MODEL_DIR / 'disease' / 'disease_mobilenet_weights.h5'
DISEASE_CLASS_INDICES_PATH = MODEL_DIR / 'disease' / 'disease_detection_class_indices.pkl'
DISEASE_PYTORCH_MODEL_PATH = MODEL_DIR / 'disease' / 'plant-disease-model.pth'

# Weather
WEATHER_MODEL_PATH = MODEL_DIR / 'weather' / 'lstm_model.h5'

# Yield
YIELD_MODEL_PATH = MODEL_DIR / 'yield' / 'crop_yield_model.pkl'
YIELD_XGBOOST_PATH = MODEL_DIR / 'yield' / 'crop_yield_xgboost.json'
YIELD_RF_PATH = MODEL_DIR / 'yield' / 'yield_predictor_model.pkl'
YIELD_AREA_ENCODER_PATH = MODEL_DIR / 'yield' / 'area_encoder.pkl'
YIELD_ITEM_ENCODER_PATH = MODEL_DIR / 'yield' / 'item_encoder.pkl'

# Price
PRICE_MODEL_PATH = MODEL_DIR / 'price' / 'prophet_model.pkl'

# Crop Recommendation
CROP_MODEL_PATH = MODEL_DIR / 'crop' / 'crop_recommendation_model.pkl'
CROP_RF_MODEL_PATH = MODEL_DIR / 'crop' / 'rf_model.pkl'
CROP_LABEL_ENCODER_PATH = MODEL_DIR / 'crop' / 'label_encoder.pkl'
CROP_SCALER_PATH = MODEL_DIR / 'crop' / 'scaler.pkl'

# Irrigation
IRRIGATION_MODEL_PATH = MODEL_DIR / 'irrigation' / 'irrigation_model.pkl'

# Fertilizer (NEW)
FERTILIZER_MODEL_PATH = MODEL_DIR / 'fertilizer' / 'fertilizer_model.pkl'
FERTILIZER_SOIL_ENCODER_PATH = MODEL_DIR / 'fertilizer' / 'soil_encoder.pkl'
FERTILIZER_CROP_ENCODER_PATH = MODEL_DIR / 'fertilizer' / 'crop_encoder.pkl'
FERTILIZER_FERTILIZER_ENCODER_PATH = MODEL_DIR / 'fertilizer' / 'fertilizer_encoder.pkl'

# Soil Classification (NEW)
SOIL_MODEL_PATH = MODEL_DIR / 'soil' / 'soil_classifier_model.h5'

# =============================================================================
# External API keys (if needed)
# =============================================================================
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
MARKET_API_KEY = os.getenv('MARKET_API_KEY', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

# =============================================================================
# Model inference settings
# =============================================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TEMPERATURE = 0.7

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'logs' / 'kalpataru.log'

# =============================================================================
# Cache settings
# =============================================================================
CACHE_DIR = BASE_DIR / '.cache'
CACHE_TTL = 3600  # seconds

# =============================================================================
# Dataset source (original location)
# =============================================================================
DATASET_SOURCE_DIR = BASE_DIR / 'Dataset'

# =============================================================================
# Supported values
# =============================================================================
SOIL_TYPES = ['Alluvial', 'Black', 'Red', 'Laterite', 'Sandy', 'Clay', 'Loamy']
SEASONS = ['Kharif', 'Rabi', 'Zaid']
REGIONS = [
    'Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 
    'Karnataka', 'Tamil Nadu', 'Gujarat', 'Madhya Pradesh',
    'Rajasthan', 'West Bengal', 'Kerala', 'Odisha', 'Assam', 'Bihar'
]
