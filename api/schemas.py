"""
API schemas for request/response validation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field

class DiseasePredictionRequest(BaseModel):
    """Schema for disease prediction request."""
    image: str = Field(..., description="Base64 encoded image")

class DiseasePredictionResponse(BaseModel):
    """Schema for disease prediction response."""
    disease: str
    confidence: float
    recommendations: List[str]

class IrrigationPredictionRequest(BaseModel):
    """Schema for irrigation prediction request."""
    soil_moisture: float = Field(..., ge=0, le=100)
    temperature: float
    humidity: float
    crop_type: str
    growth_stage: str

class WeatherPredictionRequest(BaseModel):
    """Schema for weather prediction request."""
    location: str
    historical_days: int = Field(default=30, ge=1, le=365)
    forecast_days: int = Field(default=7, ge=1, le=14)

class YieldPredictionRequest(BaseModel):
    """Schema for yield prediction request."""
    crop_type: str
    area_hectares: float = Field(..., gt=0)
    region: str
    season: str
    soil_type: str

class PricePredictionRequest(BaseModel):
    """Schema for price prediction request."""
    commodity: str
    market: str
    forecast_days: int = Field(default=7, ge=1, le=30)

class CropRecommendationRequest(BaseModel):
    """Schema for crop recommendation request."""
    soil_type: str
    rainfall_mm: float
    temperature_celsius: float
    ph_level: float = Field(..., ge=0, le=14)
    region: str
    season: str

class TranslationRequest(BaseModel):
    """Schema for translation request."""
    text: str = Field(..., min_length=1)
    target_lang: str = Field(default="en")
    source_lang: Optional[str] = Field(default=None)

class ErrorResponse(BaseModel):
    """Schema for error response."""
    success: bool = False
    message: str
    error_code: Optional[str] = None
