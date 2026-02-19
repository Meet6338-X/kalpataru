"""
Constants for Kalpataru application.
"""

# Disease classes
DISEASE_CLASSES = [
    'healthy',
    'bacterial_spot',
    'early_blight',
    'late_blight',
    'leaf_mold',
    'septoria_leaf_spot',
    'spider_mites',
    'target_spot',
    'mosaic_virus',
    'yellow_leaf_curl_virus'
]

# Crop types
CROP_TYPES = [
    'rice', 'wheat', 'maize', 'cotton', 'sugarcane',
    'potato', 'tomato', 'onion', 'mustard', 'soybean'
]

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi'
}

# Weather conditions
WEATHER_CONDITIONS = [
    'sunny', 'cloudy', 'rainy', 'stormy', 'foggy', 'snowy'
]

# Irrigation modes
IRRIGATION_MODES = ['manual', 'automatic', 'scheduled', 'smart']

# API response messages
MESSAGES = {
    'SUCCESS': 'Operation completed successfully',
    'ERROR': 'An error occurred',
    'MODEL_NOT_LOADED': 'Model not loaded',
    'INVALID_INPUT': 'Invalid input provided',
    'FILE_NOT_FOUND': 'File not found'
}

# Default values
DEFAULT_LANGUAGE = 'en'
DEFAULT_REGION = 'india'
DEFAULT_CROP = 'rice'
