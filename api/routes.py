"""
API routes for Kalpataru application.
"""

from flask import Blueprint, request, jsonify
from models.disease.inference import predict_disease
from models.irrigation.irrigation_model import predict_irrigation
from models.weather.weather_lstm import predict_weather
from models.yield.yield_xgboost import predict_yield
from models.price.price_prophet import predict_price
from models.crop.crop_recommendation import recommend_crop
from services.translation import translate_text
from utils.logger import setup_logger
from utils.helpers import create_response

api_bp = Blueprint('api', __name__)
logger = setup_logger(__name__)

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

@api_bp.route('/weather/predict', methods=['POST'])
def weather_forecast():
    """Endpoint for weather forecasting."""
    try:
        data = request.json
        result = predict_weather(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Weather prediction error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500

@api_bp.route('/yield/predict', methods=['POST'])
def yield_prediction():
    """Endpoint for yield prediction."""
    try:
        data = request.json
        result = predict_yield(data)
        
        return jsonify(create_response(True, data=result))
    except Exception as e:
        logger.error(f"Yield prediction error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500

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
