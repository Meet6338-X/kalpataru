"""
API routes for Kalpataru application.
Includes all endpoints for agricultural AI services.
"""

import importlib

from flask import Blueprint, request, jsonify
from models.disease.inference import predict_disease, get_supported_diseases
from models.irrigation.irrigation_model import predict_irrigation
from models.weather.weather_lstm import predict_weather
from models.price.price_prophet import predict_price
from models.crop.crop_recommendation import recommend_crop, get_supported_crops
from models.fertilizer.fertilizer_model import predict_fertilizer, get_fertilizer_info
from models.soil.soil_classifier import classify_soil, get_soil_info
from services.translation import translate_text
from services.risk_calculator import calculate_risk_score, calculate_insurance
from services.nutrient_service import analyze_nutrients
from services.weather_client import get_weather, get_weather_forecast
from services.ai_assistant import chat_with_ai, analyze_agricultural_image, clear_conversation
from utils.logger import setup_logger
from utils.helpers import create_response

# Import yield module using importlib (since 'yield' is a reserved keyword)
yield_model = importlib.import_module('models.yield.yield_xgboost')

api_bp = Blueprint('api', __name__)
logger = setup_logger(__name__)


# =============================================================================
# Disease Detection Endpoints
# =============================================================================

@api_bp.route('/disease/predict', methods=['POST'])
def disease_detection():
    """Endpoint for plant disease detection."""
    try:
        if 'image' not in request.files:
            return jsonify(create_response(False, message='No image provided')), 400
        
        image = request.files['image']
        result = predict_disease(image)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Disease detection error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/disease/classes', methods=['GET'])
def disease_classes():
    """Get list of supported disease classes."""
    try:
        diseases = get_supported_diseases()
        return jsonify(create_response(True, data={'diseases': diseases, 'count': len(diseases)}))
    except Exception as e:
        logger.error(f"Error getting disease classes: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Irrigation Endpoints
# =============================================================================

@api_bp.route('/irrigation/predict', methods=['POST'])
def irrigation_prediction():
    """Endpoint for irrigation prediction."""
    try:
        data = request.json
        result = predict_irrigation(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Irrigation prediction error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Weather Endpoints
# =============================================================================

@api_bp.route('/weather/predict', methods=['POST'])
def weather_forecast():
    """Endpoint for weather forecasting (legacy LSTM-based)."""
    try:
        data = request.json
        result = predict_weather(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Weather prediction error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/weather/current', methods=['GET'])
def current_weather():
    """Get current weather for a location."""
    try:
        location = request.args.get('location', 'Delhi')
        result = get_weather(location)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Weather fetch error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/weather/forecast', methods=['GET'])
def weather_forecast_api():
    """Get weather forecast for a location."""
    try:
        location = request.args.get('location', 'Delhi')
        days = int(request.args.get('days', 7))
        result = get_weather_forecast(location, days)
        
        return jsonify(create_response(True, data={'forecast': result, 'location': location}))
    except Exception as e:
        logger.error(f"Weather forecast error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Yield Prediction Endpoints
# =============================================================================

@api_bp.route('/yield/predict', methods=['POST'])
def yield_prediction():
    """Endpoint for yield prediction."""
    try:
        data = request.json
        result = yield_model.predict_yield(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Yield prediction error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/yield/crops', methods=['GET'])
def yield_supported_crops():
    """Get list of supported crops for yield prediction."""
    try:
        crops = yield_model.get_supported_crops()
        return jsonify(create_response(True, data={'crops': crops}))
    except Exception as e:
        logger.error(f"Error getting supported crops: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Price Forecasting Endpoints
# =============================================================================

@api_bp.route('/price/predict', methods=['POST'])
def price_forecast():
    """Endpoint for price forecasting."""
    try:
        data = request.json
        result = predict_price(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Price prediction error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Crop Recommendation Endpoints
# =============================================================================

@api_bp.route('/crop/recommend', methods=['POST'])
def crop_recommendation():
    """Endpoint for crop recommendation."""
    try:
        data = request.json
        result = recommend_crop(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Crop recommendation error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/crop/supported', methods=['GET'])
def supported_crops():
    """Get list of supported crops for recommendation."""
    try:
        crops = get_supported_crops()
        return jsonify(create_response(True, data={'crops': crops}))
    except Exception as e:
        logger.error(f"Error getting supported crops: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Fertilizer Recommendation Endpoints (NEW)
# =============================================================================

@api_bp.route('/fertilizer/recommend', methods=['POST'])
def fertilizer_recommendation():
    """Endpoint for fertilizer recommendation."""
    try:
        data = request.json
        result = predict_fertilizer(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Fertilizer recommendation error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/fertilizer/info/<fertilizer_name>', methods=['GET'])
def fertilizer_info(fertilizer_name):
    """Get information about a specific fertilizer."""
    try:
        result = get_fertilizer_info(fertilizer_name)
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Error getting fertilizer info: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Soil Classification Endpoints (NEW)
# =============================================================================

@api_bp.route('/soil/classify', methods=['POST'])
def soil_classification():
    """Endpoint for soil type classification from image."""
    try:
        if 'image' not in request.files:
            return jsonify(create_response(False, message='No image provided')), 400
        
        image = request.files['image']
        result = classify_soil(image)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Soil classification error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/soil/info/<soil_type>', methods=['GET'])
def soil_info(soil_type):
    """Get information about a specific soil type."""
    try:
        result = get_soil_info(soil_type)
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Error getting soil info: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Risk Calculator Endpoints (NEW)
# =============================================================================

@api_bp.route('/risk/calculate', methods=['POST'])
def risk_calculation():
    """Endpoint for calculating agricultural risk score."""
    try:
        data = request.json
        result = calculate_risk_score(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Risk calculation error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/risk/insurance', methods=['POST'])
def insurance_calculation():
    """Endpoint for calculating insurance premium."""
    try:
        data = request.json
        result = calculate_insurance(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Insurance calculation error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Nutrient Analysis Endpoints (NEW)
# =============================================================================

@api_bp.route('/nutrient/analyze', methods=['POST'])
def nutrient_analysis():
    """Endpoint for soil nutrient analysis."""
    try:
        data = request.json
        result = analyze_nutrients(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Nutrient analysis error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Translation Endpoints
# =============================================================================

@api_bp.route('/translate', methods=['POST'])
def translate():
    """Endpoint for translation."""
    try:
        data = request.json
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'en')
        
        result = translate_text(text, target_lang)
        
        return jsonify(create_response(True, data={'translated_text': result}))
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# AI Assistant Endpoints (NEW)
# =============================================================================

@api_bp.route('/ai/chat', methods=['POST'])
def ai_chat():
    """Endpoint for AI chat assistant."""
    try:
        data = request.json
        message = data.get('message', '')
        image_url = data.get('image_url')
        include_context = data.get('include_context', True)
        
        if not message:
            return jsonify(create_response(False, message='Message is required')), 400
        
        result = chat_with_ai(
            message=message,
            image_url=image_url,
            image_data=None
        )
        
        if result.get('success'):
            return jsonify(create_response(True, data={
                'response': result.get('response'),
                'timestamp': result.get('timestamp')
            }))
        else:
            return jsonify(create_response(
                False, 
                message=result.get('error', 'Unknown error'),
                data={'fallback_response': result.get('fallback_response')}
            )), 500
    except Exception as e:
        logger.error(f"AI chat error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/ai/chat/image', methods=['POST'])
def ai_chat_with_image():
    """Endpoint for AI chat with image upload."""
    try:
        message = request.form.get('message', 'Analyze this image')
        
        if 'image' not in request.files:
            return jsonify(create_response(False, message='No image provided')), 400
        
        image_file = request.files['image']
        image_data = image_file.read()
        
        result = chat_with_ai(
            message=message,
            image_data=image_data
        )
        
        if result.get('success'):
            return jsonify(create_response(True, data={
                'response': result.get('response'),
                'timestamp': result.get('timestamp')
            }))
        else:
            return jsonify(create_response(
                False,
                message=result.get('error', 'Unknown error'),
                data={'fallback_response': result.get('fallback_response')}
            )), 500
    except Exception as e:
        logger.error(f"AI chat with image error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/ai/analyze', methods=['POST'])
def ai_analyze_image():
    """Endpoint for specialized image analysis."""
    try:
        if 'image' not in request.files:
            return jsonify(create_response(False, message='No image provided')), 400
        
        image_file = request.files['image']
        analysis_type = request.form.get('analysis_type', 'general')
        image_data = image_file.read()
        
        result = analyze_agricultural_image(
            image_data=image_data,
            analysis_type=analysis_type
        )
        
        if result.get('success'):
            return jsonify(create_response(True, data={
                'analysis': result.get('response'),
                'analysis_type': analysis_type,
                'timestamp': result.get('timestamp')
            }))
        else:
            return jsonify(create_response(
                False,
                message=result.get('error', 'Unknown error'),
                data={'fallback_response': result.get('fallback_response')}
            )), 500
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/ai/clear', methods=['POST'])
def ai_clear_history():
    """Clear AI conversation history."""
    try:
        result = clear_conversation()
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# Health and Status Endpoints
# =============================================================================

@api_bp.route('/status', methods=['GET'])
def api_status():
    """Get API status and available endpoints."""
    endpoints = {
        'disease': {
            'predict': '/api/disease/predict [POST]',
            'classes': '/api/disease/classes [GET]'
        },
        'irrigation': {
            'predict': '/api/irrigation/predict [POST]'
        },
        'weather': {
            'predict': '/api/weather/predict [POST]',
            'current': '/api/weather/current?location=<city> [GET]',
            'forecast': '/api/weather/forecast?location=<city>&days=<n> [GET]'
        },
        'yield': {
            'predict': '/api/yield/predict [POST]',
            'crops': '/api/yield/crops [GET]'
        },
        'price': {
            'predict': '/api/price/predict [POST]'
        },
        'crop': {
            'recommend': '/api/crop/recommend [POST]',
            'supported': '/api/crop/supported [GET]'
        },
        'fertilizer': {
            'recommend': '/api/fertilizer/recommend [POST]',
            'info': '/api/fertilizer/info/<name> [GET]'
        },
        'soil': {
            'classify': '/api/soil/classify [POST]',
            'info': '/api/soil/info/<type> [GET]'
        },
        'risk': {
            'calculate': '/api/risk/calculate [POST]',
            'insurance': '/api/risk/insurance [POST]'
        },
        'nutrient': {
            'analyze': '/api/nutrient/analyze [POST]'
        },
        'translation': {
            'translate': '/api/translate [POST]'
        },
        'ai': {
            'chat': '/api/ai/chat [POST]',
            'chat_image': '/api/ai/chat/image [POST]',
            'analyze': '/api/ai/analyze [POST]',
            'clear': '/api/ai/clear [POST]'
        }
    }
    
    return jsonify(create_response(True, data={
        'status': 'operational',
        'version': '2.0.0',
        'endpoints': endpoints
    }))
