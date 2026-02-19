"""
Recommendation Engine Service.
"""

from typing import Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

def get_personalized_recommendations(user_data: Dict[str, Any], 
                                      predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate personalized recommendations based on user data and predictions.
    
    Args:
        user_data: User profile and context
        predictions: Model predictions
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Disease-based recommendations
    if 'disease' in predictions:
        disease = predictions.get('disease')
        if disease != 'healthy':
            recommendations.extend(get_disease_recommendations(disease))
    
    # Irrigation recommendations
    if 'irrigation' in predictions:
        irrigation = predictions.get('irrigation_needed')
        if irrigation:
            recommendations.append({
                'type': 'irrigation',
                'priority': 'high',
                'message': 'Irrigation recommended today',
                'action': 'Water crops early morning or evening'
            })
    
    # Weather-based recommendations
    if 'weather' in predictions:
        weather = predictions.get('weather', {})
        if weather.get('rainy_days', 0) > 3:
            recommendations.append({
                'type': 'weather',
                'priority': 'medium',
                'message': 'Rain expected',
                'action': 'Postpone irrigation, protect sensitive crops'
            })
    
    # Price-based recommendations
    if 'price' in predictions:
        price = predictions.get('price', {})
        trend = price.get('trend')
        if trend == 'increasing':
            recommendations.append({
                'type': 'market',
                'priority': 'medium',
                'message': 'Prices expected to rise',
                'action': 'Consider holding produce for better prices'
            })
        elif trend == 'decreasing':
            recommendations.append({
                'type': 'market',
                'priority': 'high',
                'message': 'Prices expected to fall',
                'action': 'Consider selling soon to avoid losses'
            })
    
    return recommendations

def get_disease_recommendations(disease: str) -> List[Dict[str, Any]]:
    """Get recommendations for specific disease."""
    recommendations = {
        'bacterial_spot': [
            {'type': 'treatment', 'priority': 'high', 
             'message': 'Apply copper-based fungicide', 'action': 'Spray within 24 hours'},
            {'type': 'prevention', 'priority': 'high',
             'message': 'Remove infected leaves', 'action': 'Prune and destroy affected parts'}
        ],
        'early_blight': [
            {'type': 'treatment', 'priority': 'high',
             'message': 'Apply fungicide', 'action': 'Use chlorothalonil or mancozeb'},
            {'type': 'prevention', 'priority': 'medium',
             'message': 'Improve air circulation', 'action': 'Space plants properly'}
        ],
        'late_blight': [
            {'type': 'treatment', 'priority': 'critical',
             'message': 'Remove infected plants immediately', 'action': 'Do not compost infected material'},
            {'type': 'prevention', 'priority': 'high',
             'message': 'Apply preventive fungicide', 'action': 'Use mancozeb every 7 days'}
        ]
    }
    
    return recommendations.get(disease, [
        {'type': 'general', 'priority': 'medium',
         'message': 'Consult agricultural expert', 
         'action': 'Contact local extension service'}
    ])

def get_seasonal_recommendations(season: str, region: str) -> List[Dict[str, Any]]:
    """Get seasonal recommendations."""
    seasonal = {
        'kharif': [
            'Prepare fields for monsoon planting',
            'Ensure proper drainage systems',
            'Stock up on fertilizers'
        ],
        'rabi': [
            'Prepare for winter cropping',
            'Use mulch to retain moisture',
            'Protect crops from frost'
        ],
        'zaid': [
            'Irrigation is crucial due to high temperatures',
            'Use shade nets for sensitive crops',
            'Water early morning or late evening'
        ]
    }
    
    return [{'type': 'seasonal', 'message': msg} 
            for msg in seasonal.get(season, [])]

def rank_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank recommendations by priority.
    
    Args:
        recommendations: List of recommendations
        
    Returns:
        Sorted recommendations
    """
    priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    
    return sorted(recommendations, 
                  key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
