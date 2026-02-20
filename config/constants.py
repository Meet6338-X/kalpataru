"""
Constants for Kalpataru application.
"""

# =============================================================================
# Disease Classes (38 classes from PlantVillage dataset)
# =============================================================================
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Legacy disease classes (10 classes - kept for backward compatibility)
DISEASE_CLASSES_LEGACY = [
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

# =============================================================================
# Crop Types
# =============================================================================
CROP_TYPES = [
    'rice', 'wheat', 'maize', 'cotton', 'sugarcane',
    'potato', 'tomato', 'onion', 'mustard', 'soybean',
    'groundnut', 'barley', 'millets', 'chickpea', 'paddy'
]

# =============================================================================
# Soil Types
# =============================================================================
SOIL_TYPES = [
    'Alluvial', 'Black', 'Red', 'Laterite', 
    'Sandy', 'Clay', 'Loamy'
]

# =============================================================================
# Seasons
# =============================================================================
SEASONS = ['Kharif', 'Rabi', 'Zaid']

# =============================================================================
# Fertilizer Types (NEW)
# =============================================================================
FERTILIZER_TYPES = [
    'Urea',
    'DAP',
    'MOP',
    'NPK-10-26-26',
    'NPK-20-20-20',
    'SSP',
    'Ammonium Sulphate',
    'Super Phosphate',
    '10-10-10',
    '14-35-14'
]

# =============================================================================
# Supported Languages for Translation
# =============================================================================
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

# =============================================================================
# Weather Conditions
# =============================================================================
WEATHER_CONDITIONS = [
    'sunny', 'cloudy', 'rainy', 'stormy', 'foggy', 'snowy',
    'Clear', 'Clouds', 'Rain', 'Drizzle', 'Thunderstorm'
]

# =============================================================================
# Irrigation Modes
# =============================================================================
IRRIGATION_MODES = ['manual', 'automatic', 'scheduled', 'smart']

# =============================================================================
# Growth Stages
# =============================================================================
GROWTH_STAGES = [
    'seedling', 'germination', 'vegetative', 
    'flowering', 'maturation'
]

# =============================================================================
# Regions
# =============================================================================
REGIONS = [
    'Punjab', 'Haryana', 'Uttar Pradesh', 'UP',
    'Maharashtra', 'Karnataka', 'Tamil Nadu', 'TN',
    'Gujarat', 'Madhya Pradesh', 'MP', 'Rajasthan',
    'West Bengal', 'Kerala', 'Odisha', 'Assam', 'Bihar',
    'Andhra Pradesh', 'AP', 'Telangana'
]

# =============================================================================
# Risk Categories
# =============================================================================
RISK_CATEGORIES = ['excellent', 'good', 'moderate', 'high', 'critical']

# =============================================================================
# Nutrient Status
# =============================================================================
NUTRIENT_STATUS = ['low', 'low-moderate', 'optimal', 'high', 'unknown']

# =============================================================================
# API Response Messages
# =============================================================================
MESSAGES = {
    'SUCCESS': 'Operation completed successfully',
    'ERROR': 'An error occurred',
    'MODEL_NOT_LOADED': 'Model not loaded',
    'INVALID_INPUT': 'Invalid input provided',
    'FILE_NOT_FOUND': 'File not found',
    'IMAGE_REQUIRED': 'Image file is required',
    'LOCATION_REQUIRED': 'Location is required',
    'CROP_NOT_SUPPORTED': 'Crop type not supported',
    'DISEASE_NOT_RECOGNIZED': 'Disease not recognized'
}

# =============================================================================
# Default Values
# =============================================================================
DEFAULT_LANGUAGE = 'en'
DEFAULT_REGION = 'india'
DEFAULT_CROP = 'rice'
DEFAULT_SOIL = 'alluvial'
DEFAULT_SEASON = 'kharif'
DEFAULT_TEMPERATURE = 25.0
DEFAULT_HUMIDITY = 60.0
DEFAULT_PH = 7.0
DEFAULT_RAINFALL = 1000.0

# =============================================================================
# Model Confidence Thresholds
# =============================================================================
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.70
CONFIDENCE_LOW = 0.50

# =============================================================================
# API Version
# =============================================================================
API_VERSION = '2.0.0'
