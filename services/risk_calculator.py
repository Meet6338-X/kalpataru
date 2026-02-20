"""
Risk Calculator Service for Kalpataru.
Mathematical formulas and utilities for agricultural risk assessment.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskCalculator:
    """
    Mathematical utilities for agricultural risk assessment.
    Calculates Agri-Risk Score (ARS) and insurance-related metrics.
    """
    
    # Base premium rates by crop type (percentage of coverage)
    CROP_BASE_RATES = {
        'rice': 3.5,
        'wheat': 3.0,
        'sugarcane': 4.0,
        'cotton': 4.5,
        'maize': 3.5,
        'soybean': 4.0,
        'vegetables': 5.0,
        'fruits': 5.5,
        'potato': 4.5,
        'tomato': 5.0,
        'onion': 4.5,
        'groundnut': 4.0,
        'default': 4.0
    }
    
    # Risk multipliers by ARS score range
    RISK_MULTIPLIERS = [
        (0, 20, 0.7),      # Excellent: 30% discount
        (21, 40, 0.9),     # Good: 10% discount
        (41, 60, 1.0),     # Moderate: No change
        (61, 80, 1.3),     # High: 30% increase
        (81, 100, 1.6)     # Critical: 60% increase
    ]
    
    # Weather risk thresholds
    WEATHER_THRESHOLDS = {
        'rainfall_deviation_normal': 20,  # % deviation considered normal
        'extreme_temp_days_high': 10,     # days above threshold
        'drought_days_high': 15,          # consecutive dry days
        'flood_incidents_high': 2         # flood events per season
    }
    
    @staticmethod
    def calculate_ars_score(
        weather_risk: float,
        crop_success_rate: float,
        location_risk: float,
        activity_score: float,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate Agri-Risk Score (ARS) using weighted factors.
        
        Args:
            weather_risk: Weather-related risk (0-1, higher = riskier)
            crop_success_rate: Historical crop success (0-1, higher = better)
            location_risk: Location-based risk (0-1, higher = riskier)
            activity_score: Platform activity score (0-1, higher = better)
            weights: Custom weights for each factor
            
        Returns:
            float: ARS score (0-100, lower is better)
        """
        # Default weights
        if weights is None:
            weights = {
                'weather': 0.35,
                'crop_success': 0.30,
                'location': 0.25,
                'activity': 0.10
            }
        
        # Invert positive metrics (higher is better → lower risk score)
        inverted_crop_success = 1 - crop_success_rate
        inverted_activity = 1 - activity_score
        
        # Weighted sum
        ars = (
            weather_risk * weights['weather'] +
            inverted_crop_success * weights['crop_success'] +
            location_risk * weights['location'] +
            inverted_activity * weights['activity']
        )
        
        # Scale to 0-100
        return min(100, max(0, ars * 100))
    
    @staticmethod
    def calculate_weather_risk(
        rainfall_deviation: float,
        temperature_extremes: int,
        drought_days: int,
        flood_incidents: int
    ) -> float:
        """
        Calculate weather-related risk factor.
        
        Args:
            rainfall_deviation: Deviation from normal rainfall (percentage)
            temperature_extremes: Number of extreme temperature days
            drought_days: Number of drought days in season
            flood_incidents: Number of flood events
            
        Returns:
            float: Weather risk (0-1)
        """
        # Normalize each factor
        rainfall_factor = min(1, abs(rainfall_deviation) / 50)
        temp_factor = min(1, temperature_extremes / 30)
        drought_factor = min(1, drought_days / 30)
        flood_factor = min(1, flood_incidents / 5)
        
        # Weighted combination
        weather_risk = (
            rainfall_factor * 0.3 +
            temp_factor * 0.25 +
            drought_factor * 0.25 +
            flood_factor * 0.2
        )
        
        return weather_risk
    
    @staticmethod
    def calculate_crop_success_rate(
        historical_yields: List[float],
        expected_yield: float,
        disease_incidents: int = 0,
        pest_incidents: int = 0
    ) -> float:
        """
        Calculate crop success rate based on historical data.
        
        Args:
            historical_yields: List of past yields
            expected_yield: Expected/target yield
            disease_incidents: Number of disease incidents
            pest_incidents: Number of pest incidents
            
        Returns:
            float: Success rate (0-1)
        """
        if not historical_yields:
            return 0.5  # Default if no data
        
        # Calculate yield achievement rate
        avg_yield = sum(historical_yields) / len(historical_yields)
        yield_rate = min(1, avg_yield / expected_yield) if expected_yield > 0 else 0.5
        
        # Adjust for disease and pest incidents
        health_penalty = min(0.3, (disease_incidents * 0.05) + (pest_incidents * 0.05))
        
        return max(0, yield_rate - health_penalty)
    
    @staticmethod
    def calculate_location_risk(
        soil_quality: float,
        irrigation_access: float,
        market_distance: float,
        climate_zone_risk: float = 0.5
    ) -> float:
        """
        Calculate location-based risk factor.
        
        Args:
            soil_quality: Soil quality score (0-1, higher = better)
            irrigation_access: Irrigation availability (0-1, higher = better)
            market_distance: Distance to market in km
            climate_zone_risk: Climate zone risk factor (0-1)
            
        Returns:
            float: Location risk (0-1)
        """
        # Invert positive factors
        soil_risk = 1 - soil_quality
        irrigation_risk = 1 - irrigation_access
        
        # Normalize market distance (50km = high risk)
        market_risk = min(1, market_distance / 50)
        
        # Weighted combination
        location_risk = (
            soil_risk * 0.3 +
            irrigation_risk * 0.3 +
            market_risk * 0.2 +
            climate_zone_risk * 0.2
        )
        
        return location_risk
    
    @staticmethod
    def calculate_activity_score(
        days_active: int,
        transactions: int,
        data_quality: float = 0.8
    ) -> float:
        """
        Calculate platform activity score.
        
        Args:
            days_active: Number of days of active usage
            transactions: Number of transactions recorded
            data_quality: Quality of data provided (0-1)
            
        Returns:
            float: Activity score (0-1)
        """
        # Normalize days active (365 days = full score)
        activity_rate = min(1, days_active / 365)
        
        # Normalize transactions (50 transactions = full score)
        transaction_rate = min(1, transactions / 50)
        
        # Weighted combination
        activity_score = (
            activity_rate * 0.4 +
            transaction_rate * 0.3 +
            data_quality * 0.3
        )
        
        return activity_score
    
    @classmethod
    def get_risk_multiplier(cls, ars_score: float) -> Tuple[float, str]:
        """
        Get risk multiplier and category based on ARS score.
        
        Args:
            ars_score: ARS score (0-100)
            
        Returns:
            Tuple of (multiplier, category)
        """
        for min_score, max_score, multiplier in cls.RISK_MULTIPLIERS:
            if min_score <= ars_score <= max_score:
                if ars_score <= 20:
                    category = 'excellent'
                elif ars_score <= 40:
                    category = 'good'
                elif ars_score <= 60:
                    category = 'moderate'
                elif ars_score <= 80:
                    category = 'high'
                else:
                    category = 'critical'
                return multiplier, category
        
        return 1.0, 'moderate'
    
    @classmethod
    def calculate_insurance_premium(
        cls,
        crop_type: str,
        coverage_amount: float,
        ars_score: float,
        season: str = 'kharif'
    ) -> Dict[str, Any]:
        """
        Calculate insurance premium based on risk factors.
        
        Args:
            crop_type: Type of crop
            coverage_amount: Coverage amount in rupees
            ars_score: Agri-Risk Score
            season: Growing season
            
        Returns:
            Dictionary with premium details
        """
        # Get base rate
        base_rate = cls.CROP_BASE_RATES.get(
            crop_type.lower(), 
            cls.CROP_BASE_RATES['default']
        )
        
        # Get risk multiplier
        risk_multiplier, risk_category = cls.get_risk_multiplier(ars_score)
        
        # Season adjustment
        season_factors = {
            'kharif': 1.0,
            'rabi': 0.95,
            'zaid': 1.05
        }
        season_factor = season_factors.get(season.lower(), 1.0)
        
        # Calculate premium
        adjusted_rate = base_rate * risk_multiplier * season_factor
        premium = (coverage_amount * adjusted_rate) / 100
        
        return {
            'premium_amount': round(premium, 2),
            'base_rate': base_rate,
            'adjusted_rate': round(adjusted_rate, 2),
            'risk_multiplier': risk_multiplier,
            'risk_category': risk_category,
            'season_factor': season_factor,
            'coverage_amount': coverage_amount,
            'sum_insured': coverage_amount
        }
    
    @staticmethod
    def calculate_payout(
        coverage_amount: float,
        actual_yield: float,
        expected_yield: float,
        threshold_percentage: float = 70.0
    ) -> Dict[str, Any]:
        """
        Calculate insurance payout based on yield loss.
        
        Args:
            coverage_amount: Coverage amount
            actual_yield: Actual yield achieved
            expected_yield: Expected/target yield
            threshold_percentage: Loss threshold for payout
            
        Returns:
            Dictionary with payout details
        """
        # Calculate yield percentage
        yield_percentage = (actual_yield / expected_yield * 100) if expected_yield > 0 else 100
        
        # Check if below threshold
        if yield_percentage >= threshold_percentage:
            return {
                'payout_amount': 0,
                'eligible': False,
                'reason': f'Yield ({yield_percentage:.1f}%) above threshold ({threshold_percentage}%)',
                'yield_percentage': round(yield_percentage, 2)
            }
        
        # Calculate loss percentage
        loss_percentage = threshold_percentage - yield_percentage
        
        # Calculate payout (proportional to loss)
        payout = (coverage_amount * loss_percentage) / 100
        
        return {
            'payout_amount': round(payout, 2),
            'eligible': True,
            'loss_percentage': round(loss_percentage, 2),
            'yield_percentage': round(yield_percentage, 2),
            'threshold_percentage': threshold_percentage
        }


def calculate_risk_score(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to calculate comprehensive risk score.
    
    Args:
        data: Dictionary containing risk factors:
            - weather_risk: float (0-1)
            - crop_success_rate: float (0-1)
            - location_risk: float (0-1)
            - activity_score: float (0-1)
            - weights: Dict[str, float] (optional)
            
    Returns:
        Risk assessment dictionary
    """
    calculator = RiskCalculator()
    
    ars_score = calculator.calculate_ars_score(
        weather_risk=data.get('weather_risk', 0.5),
        crop_success_rate=data.get('crop_success_rate', 0.7),
        location_risk=data.get('location_risk', 0.3),
        activity_score=data.get('activity_score', 0.6),
        weights=data.get('weights')
    )
    
    multiplier, category = calculator.get_risk_multiplier(ars_score)
    
    return {
        'ars_score': round(ars_score, 2),
        'risk_category': category,
        'risk_multiplier': multiplier,
        'components': {
            'weather_risk': data.get('weather_risk', 0.5),
            'crop_success_rate': data.get('crop_success_rate', 0.7),
            'location_risk': data.get('location_risk', 0.3),
            'activity_score': data.get('activity_score', 0.6)
        },
        'recommendations': _get_risk_recommendations(ars_score, category)
    }


def calculate_insurance(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to calculate insurance premium.
    
    Args:
        data: Dictionary containing:
            - crop_type: str
            - coverage_amount: float
            - ars_score: float
            - season: str (optional)
            
    Returns:
        Insurance premium details
    """
    calculator = RiskCalculator()
    
    return calculator.calculate_insurance_premium(
        crop_type=data.get('crop_type', 'rice'),
        coverage_amount=data.get('coverage_amount', 50000),
        ars_score=data.get('ars_score', 50),
        season=data.get('season', 'kharif')
    )


def _get_risk_recommendations(ars_score: float, category: str) -> List[str]:
    """Get recommendations based on risk score."""
    recommendations = []
    
    if category == 'critical':
        recommendations.extend([
            'Consider crop insurance immediately',
            'Implement water conservation measures',
            'Diversify crops to spread risk',
            'Seek expert consultation for risk mitigation'
        ])
    elif category == 'high':
        recommendations.extend([
            'Monitor weather conditions closely',
            'Ensure adequate irrigation backup',
            'Consider partial crop insurance',
            'Implement pest and disease prevention measures'
        ])
    elif category == 'moderate':
        recommendations.extend([
            'Continue regular monitoring',
            'Maintain good agricultural practices',
            'Keep records for better risk assessment'
        ])
    else:
        recommendations.extend([
            'Maintain current practices',
            'Consider expanding cultivation',
            'Share best practices with community'
        ])
    
    return recommendations
