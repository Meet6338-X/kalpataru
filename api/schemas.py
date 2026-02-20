"""
API schemas for request/response validation.
Pydantic models for all API endpoints.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Disease Detection Schemas
# =============================================================================

class DiseasePredictionRequest(BaseModel):
    """Schema for disease prediction request."""
    image: str = Field(..., description="Base64 encoded image")


class DiseasePredictionResponse(BaseModel):
    """Schema for disease prediction response."""
    disease: str
    confidence: float
    is_healthy: bool
    description: str
    treatment: str
    prevention: str
    recommendations: List[str]
    model_type: str


# =============================================================================
# Irrigation Schemas
# =============================================================================

class IrrigationPredictionRequest(BaseModel):
    """Schema for irrigation prediction request."""
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture percentage")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    crop_type: str = Field(..., description="Type of crop")
    growth_stage: str = Field(default="vegetative", description="Growth stage of crop")


class IrrigationPredictionResponse(BaseModel):
    """Schema for irrigation prediction response."""
    water_requirement_mm: float
    irrigation_needed: bool
    recommended_schedule: Dict[str, Any]
    confidence: float


# =============================================================================
# Weather Schemas
# =============================================================================

class WeatherPredictionRequest(BaseModel):
    """Schema for weather prediction request."""
    location: str = Field(..., description="Location name")
    historical_days: int = Field(default=30, ge=1, le=365, description="Historical data days")
    forecast_days: int = Field(default=7, ge=1, le=14, description="Forecast days")


class WeatherCurrentRequest(BaseModel):
    """Schema for current weather request."""
    location: str = Field(default="Delhi", description="Location name")


class WeatherForecastRequest(BaseModel):
    """Schema for weather forecast request."""
    location: str = Field(default="Delhi", description="Location name")
    days: int = Field(default=7, ge=1, le=14, description="Number of forecast days")


# =============================================================================
# Yield Prediction Schemas
# =============================================================================

class YieldPredictionRequest(BaseModel):
    """Schema for yield prediction request."""
    crop_type: str = Field(..., description="Type of crop")
    area_hectares: float = Field(..., gt=0, description="Area in hectares")
    region: Optional[str] = Field(default=None, description="Geographic region")
    season: Optional[str] = Field(default=None, description="Growing season")
    soil_type: Optional[str] = Field(default=None, description="Type of soil")
    rainfall: Optional[float] = Field(default=None, ge=0, description="Annual rainfall in mm")
    temperature: Optional[float] = Field(default=None, description="Average temperature")
    pesticide: Optional[float] = Field(default=None, ge=0, description="Pesticide usage")


class YieldPredictionResponse(BaseModel):
    """Schema for yield prediction response."""
    crop_type: str
    area_hectares: float
    predicted_yield_quintals: float
    yield_per_hectare: float
    confidence: float
    prediction_method: str
    recommendations: List[str]


# =============================================================================
# Price Prediction Schemas
# =============================================================================

class PricePredictionRequest(BaseModel):
    """Schema for price prediction request."""
    commodity: str = Field(..., description="Commodity name")
    market: str = Field(default="delhi", description="Market name")
    forecast_days: int = Field(default=7, ge=1, le=30, description="Forecast days")


class PricePredictionResponse(BaseModel):
    """Schema for price prediction response."""
    commodity: str
    market: str
    current_price: float
    forecast: List[Dict[str, Any]]
    trend: str
    confidence: float


# =============================================================================
# Crop Recommendation Schemas
# =============================================================================

class CropRecommendationRequest(BaseModel):
    """Schema for crop recommendation request."""
    N: float = Field(..., ge=0, description="Nitrogen level")
    P: float = Field(..., ge=0, description="Phosphorus level")
    K: float = Field(..., ge=0, description="Potassium level")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")
    soil_type: Optional[str] = Field(default=None, description="Type of soil")
    season: Optional[str] = Field(default=None, description="Growing season")


class CropRecommendationResponse(BaseModel):
    """Schema for crop recommendation response."""
    best_crop: str
    confidence: float
    recommendations: List[Dict[str, Any]]
    method: str


# =============================================================================
# Fertilizer Recommendation Schemas (NEW)
# =============================================================================

class FertilizerRecommendationRequest(BaseModel):
    """Schema for fertilizer recommendation request."""
    temperature: float = Field(default=25.0, description="Temperature in Celsius")
    humidity: float = Field(default=60.0, ge=0, le=100, description="Humidity percentage")
    moisture: float = Field(default=50.0, ge=0, le=100, description="Soil moisture percentage")
    soil_type: str = Field(default="Loamy", description="Type of soil")
    crop_type: str = Field(default="Wheat", description="Type of crop")
    nitrogen: int = Field(default=50, ge=0, le=200, description="Nitrogen level")
    potassium: int = Field(default=40, ge=0, le=200, description="Potassium level")
    phosphorous: int = Field(default=30, ge=0, le=200, description="Phosphorous level")


class FertilizerRecommendationResponse(BaseModel):
    """Schema for fertilizer recommendation response."""
    recommended_fertilizer: str
    fertilizer_type: str
    npk_ratio: Dict[str, int]
    application_method: str
    application_notes: str
    nutrient_analysis: Dict[str, Any]
    confidence: float


# =============================================================================
# Soil Classification Schemas (NEW)
# =============================================================================

class SoilClassificationRequest(BaseModel):
    """Schema for soil classification request."""
    image: str = Field(..., description="Base64 encoded soil image")


class SoilClassificationResponse(BaseModel):
    """Schema for soil classification response."""
    soil_type: str
    confidence: float
    characteristics: List[str]
    recommended_crops: List[str]
    ph_range: List[float]
    texture: str
    drainage: str
    method: str


# =============================================================================
# Risk Calculator Schemas (NEW)
# =============================================================================

class RiskCalculationRequest(BaseModel):
    """Schema for risk calculation request."""
    weather_risk: float = Field(default=0.5, ge=0, le=1, description="Weather risk factor")
    crop_success_rate: float = Field(default=0.7, ge=0, le=1, description="Crop success rate")
    location_risk: float = Field(default=0.3, ge=0, le=1, description="Location risk factor")
    activity_score: float = Field(default=0.6, ge=0, le=1, description="Activity score")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Custom weights")


class RiskCalculationResponse(BaseModel):
    """Schema for risk calculation response."""
    ars_score: float
    risk_category: str
    risk_multiplier: float
    components: Dict[str, float]
    recommendations: List[str]


class InsuranceCalculationRequest(BaseModel):
    """Schema for insurance calculation request."""
    crop_type: str = Field(..., description="Type of crop")
    coverage_amount: float = Field(..., gt=0, description="Coverage amount in rupees")
    ars_score: float = Field(..., ge=0, le=100, description="Agri-Risk Score")
    season: str = Field(default="kharif", description="Growing season")


class InsuranceCalculationResponse(BaseModel):
    """Schema for insurance calculation response."""
    premium_amount: float
    base_rate: float
    adjusted_rate: float
    risk_multiplier: float
    risk_category: str
    season_factor: float
    coverage_amount: float
    sum_insured: float


# =============================================================================
# Nutrient Analysis Schemas (NEW)
# =============================================================================

class NutrientAnalysisRequest(BaseModel):
    """Schema for nutrient analysis request."""
    N: float = Field(default=50, ge=0, description="Nitrogen level (kg/ha)")
    P: float = Field(default=30, ge=0, description="Phosphorus level (kg/ha)")
    K: float = Field(default=40, ge=0, description="Potassium level (kg/ha)")
    crop_type: str = Field(default="wheat", description="Target crop")
    area_hectares: float = Field(default=1.0, gt=0, description="Area in hectares")
    pH: Optional[float] = Field(default=None, ge=0, le=14, description="Soil pH")
    secondary: Optional[Dict[str, float]] = Field(default=None, description="Secondary nutrients")
    micro: Optional[Dict[str, float]] = Field(default=None, description="Micronutrients")


class NutrientAnalysisResponse(BaseModel):
    """Schema for nutrient analysis response."""
    crop: str
    area_hectares: float
    current_levels: Dict[str, Dict[str, Any]]
    target_levels: Dict[str, float]
    nutrient_gaps: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    summary: str


# =============================================================================
# Translation Schemas
# =============================================================================

class TranslationRequest(BaseModel):
    """Schema for translation request."""
    text: str = Field(..., min_length=1, description="Text to translate")
    target_lang: str = Field(default="en", description="Target language code")
    source_lang: Optional[str] = Field(default=None, description="Source language code")


class TranslationResponse(BaseModel):
    """Schema for translation response."""
    translated_text: str


# =============================================================================
# Error Response Schema
# =============================================================================

class ErrorResponse(BaseModel):
    """Schema for error response."""
    success: bool = False
    message: str
    error_code: Optional[str] = None


# =============================================================================
# Generic Response Schema
# =============================================================================

class ApiResponse(BaseModel):
    """Generic API response schema."""
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
