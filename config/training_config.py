"""
Training Configuration for Kalpataru Models.

This module contains all training-related configuration parameters.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

# Data paths
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
ANALYSIS_DIR = DATA_DIR / 'analysis'

# Model paths
CROP_MODEL_DIR = MODEL_DIR / 'crop'
YIELD_MODEL_DIR = MODEL_DIR / 'yield'
DISEASE_MODEL_DIR = MODEL_DIR / 'disease'
WEATHER_MODEL_DIR = MODEL_DIR / 'weather'
PRICE_MODEL_DIR = MODEL_DIR / 'price'
IRRIGATION_MODEL_DIR = MODEL_DIR / 'irrigation'

# =============================================================================
# Crop Recommendation Model Configuration
# =============================================================================
CROP_CONFIG = {
    'model_type': 'random_forest',  # Options: 'random_forest', 'xgboost', 'gradient_boosting'
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'test_size': 0.2,
    'random_state': 42
}

# =============================================================================
# Crop Yield Model Configuration
# =============================================================================
YIELD_CONFIG = {
    'model_type': 'xgboost',  # Options: 'xgboost', 'random_forest', 'gradient_boosting', 'ridge'
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'ridge': {
        'alpha': 1.0,
        'random_state': 42
    },
    'test_size': 0.2,
    'random_state': 42
}

# =============================================================================
# Disease Detection Model Configuration
# =============================================================================
DISEASE_CONFIG = {
    'model_type': 'transfer_learning',  # Options: 'custom_cnn', 'transfer_learning'
    'base_model': 'mobilenet',  # Options: 'mobilenet', 'efficientnet', 'resnet'
    
    # Image settings
    'image_size': (224, 224),
    'color_mode': 'rgb',
    
    # Training settings
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'fine_tune_epochs': 20,
    'fine_tune_layers': 20,
    'fine_tune_learning_rate': 0.0001,
    
    # Data augmentation
    'augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    },
    
    # Data split
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    
    # Callbacks
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7
}

# =============================================================================
# Weather Model Configuration
# =============================================================================
WEATHER_CONFIG = {
    'model_type': 'lstm',
    'sequence_length': 7,
    'lstm_units': [64, 32],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# =============================================================================
# Price Model Configuration
# =============================================================================
PRICE_CONFIG = {
    'model_type': 'prophet',
    'forecast_days': 30,
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False
}

# =============================================================================
# Irrigation Model Configuration
# =============================================================================
IRRIGATION_CONFIG = {
    'model_type': 'regression',
    'features': ['soil_moisture', 'temperature', 'humidity', 'crop_type', 'growth_stage'],
    'test_size': 0.2,
    'random_state': 42
}

# =============================================================================
# Logging Configuration
# =============================================================================
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': str(BASE_DIR / 'logs' / 'training.log'),
            'mode': 'a'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}