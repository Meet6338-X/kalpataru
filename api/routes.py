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
from api.mandi import mandi_bp
from utils.logger import setup_logger
from utils.helpers import create_response
import uuid
import datetime

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
    """Endpoint for specialized image analysis.
    
    For disease analysis, runs ML model first and includes prediction results.
    Returns both LLM analysis and ML prediction data (if applicable).
    """
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
            response_data = {
                'analysis': result.get('response'),
                'analysis_type': analysis_type,
                'timestamp': result.get('timestamp')
            }
            
            # Include ML prediction results if available (for disease analysis)
            if 'ml_prediction' in result:
                response_data['ml_prediction'] = result['ml_prediction']
            
            return jsonify(create_response(True, data=response_data))
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


# =============================================================================
# FarmLink Endpoints (In-Memory Storage)
# =============================================================================
# Sample data for demo purposes
farmlink_listings = [
    {
        'id': 1,
        'listing_id': 'FL-A1B2C3D4',
        'farmer_id': 1,
        'farmer_name': 'Ramesh Kumar',
        'crop_name': 'Basmati Rice',
        'quantity_tons': 50,
        'price_per_ton': 45000,
        'location': 'Nashik',
        'state': 'Maharashtra',
        'description': 'Premium quality basmati rice, ready for immediate delivery.',
        'image_url': '',
        'status': 'ACTIVE',
        'created_at': '2026-02-20T10:00:00'
    },
    {
        'id': 2,
        'listing_id': 'FL-E5F6G7H8',
        'farmer_id': 2,
        'farmer_name': 'Suresh Patil',
        'crop_name': 'Tomatoes',
        'quantity_tons': 25,
        'price_per_ton': 18000,
        'location': 'Bangalore',
        'state': 'Karnataka',
        'description': 'Fresh tomatoes, Grade A quality, harvested yesterday.',
        'image_url': '',
        'status': 'ACTIVE',
        'created_at': '2026-02-21T08:00:00'
    },
    {
        'id': 3,
        'listing_id': 'FL-I9J0K1L2',
        'farmer_id': 3,
        'farmer_name': 'Gurpreet Singh',
        'crop_name': 'Potatoes',
        'quantity_tons': 100,
        'price_per_ton': 12000,
        'location': 'Amritsar',
        'state': 'Punjab',
        'description': 'Quality potatoes, suitable for chips and storage.',
        'image_url': '',
        'status': 'ACTIVE',
        'created_at': '2026-02-19T14:00:00'
    },
    {
        'id': 4,
        'listing_id': 'FL-M3N4O5P6',
        'farmer_id': 4,
        'farmer_name': 'Ajay Sharma',
        'crop_name': 'Onions',
        'quantity_tons': 40,
        'price_per_ton': 22000,
        'location': 'Ahmednagar',
        'state': 'Maharashtra',
        'description': 'Red onions, good quality, immediate delivery available.',
        'image_url': '',
        'status': 'ACTIVE',
        'created_at': '2026-02-21T06:00:00'
    }
]

# Sample offers from other users on listing 1 (farmer_id=1's listing)
farmlink_offers = [
    {
        'id': 1,
        'offer_id': 'OF-12345678',
        'listing_id': 1,
        'buyer_id': 2,
        'buyer_name': 'Suresh Patil',
        'offered_price': 44000,
        'message': 'Interested in bulk purchase. Can you offer discount for 50 tons?',
        'status': 'PENDING',
        'created_at': '2026-02-21T10:00:00'
    },
    {
        'id': 2,
        'offer_id': 'OF-87654321',
        'listing_id': 1,
        'buyer_id': 3,
        'buyer_name': 'Gurpreet Singh',
        'offered_price': 42000,
        'message': 'Looking for immediate delivery. Can pay upfront.',
        'status': 'PENDING',
        'created_at': '2026-02-21T11:00:00'
    }
]
farmlink_deals = []

# Sample messages for conversation demo
farmlink_messages = [
    {
        'id': 1,
        'conversation_id': '2-1-2',
        'listing_id': 2,
        'sender_id': 1,
        'receiver_id': 2,
        'message': 'Hi! I am interested in your tomatoes. What is the best price you can offer?',
        'is_read': True,
        'created_at': '2026-02-21T10:00:00'
    },
    {
        'id': 2,
        'conversation_id': '2-1-2',
        'listing_id': 2,
        'sender_id': 2,
        'receiver_id': 1,
        'message': 'Hello! For 25 tons, I can offer ₹17,000 per ton. Are you looking for immediate delivery?',
        'is_read': True,
        'created_at': '2026-02-21T10:05:00'
    },
    {
        'id': 3,
        'conversation_id': '2-1-2',
        'listing_id': 2,
        'sender_id': 1,
        'receiver_id': 2,
        'message': 'That sounds good! Yes, I need immediate delivery. Can we close the deal?',
        'is_read': True,
        'created_at': '2026-02-21T10:10:00'
    },
    {
        'id': 4,
        'conversation_id': '2-1-2',
        'listing_id': 2,
        'sender_id': 2,
        'receiver_id': 1,
        'message': 'Sure! I will prepare the produce. You can make an offer on my listing to proceed.',
        'is_read': False,
        'created_at': '2026-02-21T10:15:00'
    }
]


@api_bp.route('/farmlink/listings', methods=['POST'])
def create_farmlink_listing():
    """Create a new produce listing."""
    try:
        data = request.get_json()
        
        if not data.get('crop_name') or not data.get('quantity_tons') or not data.get('price_per_ton'):
            return jsonify(create_response(False, message='Missing required fields')), 400
        
        listing_id = f'FL-{uuid.uuid4().hex[:8].upper()}'
        listing = {
            'id': len(farmlink_listings) + 1,
            'listing_id': listing_id,
            'farmer_id': 1,
            'farmer_name': 'You',
            'crop_name': data.get('crop_name'),
            'quantity_tons': float(data.get('quantity_tons')),
            'price_per_ton': float(data.get('price_per_ton')),
            'location': data.get('location', ''),
            'state': data.get('state', ''),
            'description': data.get('description', ''),
            'image_url': data.get('image_url', ''),
            'status': 'ACTIVE',
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        
        farmlink_listings.append(listing)
        
        return jsonify(create_response(True, data=listing)), 201
    except Exception as e:
        logger.error(f"FarmLink listing error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/listings', methods=['GET'])
def get_farmlink_listings():
    """Get all active listings."""
    try:
        crop = request.args.get('crop', '').lower()
        state = request.args.get('state', '').lower()
        
        filtered = [l for l in farmlink_listings if l['status'] == 'ACTIVE']
        
        if crop:
            filtered = [l for l in filtered if crop in l['crop_name'].lower()]
        if state:
            filtered = [l for l in filtered if state in l['state'].lower()]
        
        return jsonify(create_response(True, data={
            'listings': filtered,
            'count': len(filtered)
        }))
    except Exception as e:
        logger.error(f"FarmLink listings error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/listings/<int:listing_id>', methods=['GET'])
def get_farmlink_listing(listing_id):
    """Get a single listing."""
    try:
        listing = next((l for l in farmlink_listings if l['id'] == listing_id), None)
        
        if not listing:
            return jsonify(create_response(False, message='Listing not found')), 404
        
        return jsonify(create_response(True, data=listing))
    except Exception as e:
        logger.error(f"FarmLink listing error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/my-listings', methods=['GET'])
def get_my_farmlink_listings():
    """Get current user's listings."""
    try:
        # For demo, user_id = 1
        user_listings = [l for l in farmlink_listings if l['farmer_id'] == 1]
        return jsonify(create_response(True, data={
            'listings': user_listings,
            'count': len(user_listings)
        }))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/listings/<int:listing_id>/offers', methods=['POST'])
def create_farmlink_offer(listing_id):
    """Create an offer on a listing."""
    try:
        data = request.get_json()
        listing = next((l for l in farmlink_listings if l['id'] == listing_id), None)
        
        if not listing:
            return jsonify(create_response(False, message='Listing not found')), 404
        
        if listing['status'] != 'ACTIVE':
            return jsonify(create_response(False, message='Listing is not active')), 400
        
        offer_id = f'OF-{uuid.uuid4().hex[:8].upper()}'
        offer = {
            'id': len(farmlink_offers) + 1,
            'offer_id': offer_id,
            'listing_id': listing_id,
            'buyer_id': 1,
            'buyer_name': 'You',
            'offered_price': float(data.get('offered_price', 0)),
            'message': data.get('message', ''),
            'status': 'PENDING',
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        
        farmlink_offers.append(offer)
        
        return jsonify(create_response(True, data=offer)), 201
    except Exception as e:
        logger.error(f"FarmLink offer error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/listings/<int:listing_id>/offers', methods=['GET'])
def get_listing_offers(listing_id):
    """Get offers for a listing."""
    try:
        offers = [o for o in farmlink_offers if o['listing_id'] == listing_id]
        return jsonify(create_response(True, data={
            'offers': offers,
            'count': len(offers)
        }))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/offers/<int:offer_id>/accept', methods=['POST'])
def accept_farmlink_offer(offer_id):
    """Accept an offer."""
    try:
        offer = next((o for o in farmlink_offers if o['id'] == offer_id), None)
        
        if not offer:
            return jsonify(create_response(False, message='Offer not found')), 404
        
        if offer['status'] != 'PENDING':
            return jsonify(create_response(False, message='Offer is not pending')), 400
        
        listing = next((l for l in farmlink_listings if l['id'] == offer['listing_id']), None)
        
        # Create deal
        deal_id = f'DL-{uuid.uuid4().hex[:8].upper()}'
        deal = {
            'id': len(farmlink_deals) + 1,
            'deal_id': deal_id,
            'listing_id': listing['id'],
            'offer_id': offer['id'],
            'farmer_id': listing['farmer_id'],
            'farmer_name': listing['farmer_name'],
            'buyer_id': offer['buyer_id'],
            'buyer_name': offer['buyer_name'],
            'crop_name': listing['crop_name'],
            'final_price': offer['offered_price'],
            'quantity_tons': listing['quantity_tons'],
            'total_value': offer['offered_price'] * listing['quantity_tons'],
            'status': 'PENDING',
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        
        farmlink_deals.append(deal)
        
        # Update offer with deal reference
        offer['status'] = 'ACCEPTED'
        offer['deal_id'] = deal['id']
        
        # Update listing
        listing['status'] = 'SOLD'
        
        return jsonify(create_response(True, data={'deal': deal, 'offer': offer}))
    except Exception as e:
        logger.error(f"FarmLink accept error: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/my-offers', methods=['GET'])
def get_my_farmlink_offers():
    """Get current user's offers with listing details."""
    try:
        user_offers = [o for o in farmlink_offers if o['buyer_id'] == 1]
        # Enrich offers with listing details
        enriched_offers = []
        for offer in user_offers:
            listing = next((l for l in farmlink_listings if l['id'] == offer['listing_id']), None)
            enriched_offer = offer.copy()
            if listing:
                enriched_offer['crop_name'] = listing.get('crop_name', 'Unknown')
                enriched_offer['farmer_id'] = listing.get('farmer_id')
                enriched_offer['farmer_name'] = listing.get('farmer_name', 'Farmer')
                enriched_offer['quantity_tons'] = listing.get('quantity_tons', 0)
                enriched_offer['listing_price'] = listing.get('price_per_ton', 0)
            # Add deal_id if offer was accepted
            if offer.get('deal_id'):
                enriched_offer['deal_id'] = offer['deal_id']
            enriched_offers.append(enriched_offer)
        return jsonify(create_response(True, data={
            'offers': enriched_offers,
            'count': len(enriched_offers)
        }))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/deals', methods=['GET'])
def get_farmlink_deals():
    """Get user's deals."""
    try:
        user_deals = [d for d in farmlink_deals if d['farmer_id'] == 1 or d['buyer_id'] == 1]
        return jsonify(create_response(True, data={
            'deals': user_deals,
            'count': len(user_deals)
        }))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


# ============================================
# FarmLink Additional Endpoints
# ============================================

@api_bp.route('/farmlink/listings/<int:listing_id>', methods=['PUT'])
def update_farmlink_listing(listing_id):
    """Update a listing."""
    try:
        data = request.get_json()
        
        # Find the listing
        listing = None
        for l in farmlink_listings:
            if l['id'] == listing_id:
                listing = l
                break
        
        if not listing:
            return jsonify(create_response(False, message='Listing not found')), 404
        
        # Update allowed fields
        updatable_fields = ['title', 'description', 'quantity', 'unit', 'price_per_unit', 
                          'category', 'location', 'harvest_date', 'min_order', 'images']
        for field in updatable_fields:
            if field in data:
                listing[field] = data[field]
        
        listing['updated_at'] = datetime.now().isoformat()
        
        return jsonify(create_response(True, data={'listing': listing}, message='Listing updated successfully'))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/listings/<int:listing_id>', methods=['DELETE'])
def delete_farmlink_listing(listing_id):
    """Delete a listing."""
    global farmlink_listings
    try:
        # Find and remove the listing
        listing_index = None
        for i, l in enumerate(farmlink_listings):
            if l['id'] == listing_id:
                listing_index = i
                break
        
        if listing_index is None:
            return jsonify(create_response(False, message='Listing not found')), 404
        
        # Check if there are active deals for this listing
        active_deals = [d for d in farmlink_deals if d['listing_id'] == listing_id and d['status'] == 'active']
        if active_deals:
            return jsonify(create_response(False, message='Cannot delete listing with active deals')), 400
        
        # Remove the listing
        deleted_listing = farmlink_listings.pop(listing_index)
        
        # Also remove any pending offers for this listing
        global farmlink_offers
        farmlink_offers = [o for o in farmlink_offers if o['listing_id'] != listing_id]
        
        return jsonify(create_response(True, message='Listing deleted successfully'))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/offers/<int:offer_id>/reject', methods=['POST'])
def reject_farmlink_offer(offer_id):
    """Reject an offer."""
    try:
        # Find the offer
        offer = None
        for o in farmlink_offers:
            if o['id'] == offer_id:
                offer = o
                break
        
        if not offer:
            return jsonify(create_response(False, message='Offer not found')), 404
        
        if offer['status'] != 'pending':
            return jsonify(create_response(False, message='Offer is no longer pending')), 400
        
        # Update offer status
        offer['status'] = 'rejected'
        offer['rejected_at'] = datetime.now().isoformat()
        
        return jsonify(create_response(True, data={'offer': offer}, message='Offer rejected successfully'))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/deals/<int:deal_id>', methods=['PUT'])
def update_farmlink_deal(deal_id):
    """Update deal status."""
    try:
        data = request.get_json()
        new_status = data.get('status', '').upper()  # Normalize to uppercase
        
        # Validate status
        valid_statuses = ['PENDING', 'ACTIVE', 'COMPLETED', 'CANCELLED', 'IN_TRANSIT', 'DELIVERED']
        if not new_status or new_status not in valid_statuses:
            return jsonify(create_response(False, message=f'Invalid status. Must be one of: {", ".join(valid_statuses)}')), 400
        
        # Find the deal
        deal = None
        for d in farmlink_deals:
            if d['id'] == deal_id:
                deal = d
                break
        
        if not deal:
            return jsonify(create_response(False, message='Deal not found')), 404
        
        # Update deal status
        old_status = deal['status']
        deal['status'] = new_status
        deal['updated_at'] = datetime.datetime.utcnow().isoformat()
        
        # Add status history
        if 'status_history' not in deal:
            deal['status_history'] = []
        deal['status_history'].append({
            'from': old_status,
            'to': new_status,
            'timestamp': datetime.datetime.utcnow().isoformat()
        })
        
        return jsonify(create_response(True, data={'deal': deal}, message=f'Deal status updated to {new_status}'))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/offers/<int:offer_id>', methods=['GET'])
def get_farmlink_offer(offer_id):
    """Get a specific offer."""
    try:
        offer = next((o for o in farmlink_offers if o['id'] == offer_id), None)
        if not offer:
            return jsonify(create_response(False, message='Offer not found')), 404
        
        # Get associated listing
        listing = next((l for l in farmlink_listings if l['id'] == offer['listing_id']), None)
        
        return jsonify(create_response(True, data={
            'offer': offer,
            'listing': listing
        }))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/deals/<int:deal_id>', methods=['GET'])
def get_farmlink_deal(deal_id):
    """Get a specific deal."""
    try:
        deal = next((d for d in farmlink_deals if d['id'] == deal_id), None)
        if not deal:
            return jsonify(create_response(False, message='Deal not found')), 404
        
        # Get associated listing and offer
        listing = next((l for l in farmlink_listings if l['id'] == deal['listing_id']), None)
        offer = next((o for o in farmlink_offers if o['id'] == deal['offer_id']), None)
        
        return jsonify(create_response(True, data={
            'deal': deal,
            'listing': listing,
            'offer': offer
        }))
    except Exception as e:
        return jsonify(create_response(False, message=str(e))), 500


# =============================================================================
# FarmLink Messaging Endpoints
# =============================================================================

@api_bp.route('/farmlink/messages', methods=['POST'])
def send_farmlink_message():
    """Send a message to another user about a listing."""
    try:
        data = request.get_json()
        listing_id = data.get('listing_id')
        receiver_id = data.get('receiver_id')
        message_text = data.get('message')
        
        if not all([listing_id, receiver_id, message_text]):
            return jsonify(create_response(False, message='listing_id, receiver_id, and message are required')), 400
        
        # Verify listing exists
        listing = next((l for l in farmlink_listings if l['id'] == listing_id), None)
        if not listing:
            return jsonify(create_response(False, message='Listing not found')), 404
        
        # For demo, sender_id = 1
        sender_id = 1
        
        # Create conversation ID (consistent format)
        conversation_id = f"{listing_id}-{min(sender_id, receiver_id)}-{max(sender_id, receiver_id)}"
        
        message = {
            'id': len(farmlink_messages) + 1,
            'conversation_id': conversation_id,
            'listing_id': listing_id,
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'message': message_text,
            'is_read': False,
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        
        farmlink_messages.append(message)
        
        return jsonify(create_response(True, data={'message': message}))
    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/conversations', methods=['GET'])
def get_farmlink_conversations():
    """Get all conversations for the current user."""
    try:
        # For demo, user_id = 1
        user_id = 1
        
        # Get all messages involving this user
        user_messages = [m for m in farmlink_messages if m['sender_id'] == user_id or m['receiver_id'] == user_id]
        
        # Group by conversation_id
        conversations = {}
        for msg in user_messages:
            conv_id = msg['conversation_id']
            if conv_id not in conversations:
                # Get listing info
                listing = next((l for l in farmlink_listings if l['id'] == msg['listing_id']), None)
                # Determine other user
                other_user_id = msg['receiver_id'] if msg['sender_id'] == user_id else msg['sender_id']
                
                conversations[conv_id] = {
                    'conversation_id': conv_id,
                    'listing_id': msg['listing_id'],
                    'listing_title': listing['crop_name'] if listing else 'Unknown',
                    'other_user_id': other_user_id,
                    'other_user_name': f"User {other_user_id}",
                    'last_message': msg['message'],
                    'last_message_at': msg['created_at'],
                    'unread_count': 0
                }
            
            # Count unread messages
            if msg['receiver_id'] == user_id and not msg['is_read']:
                conversations[conv_id]['unread_count'] += 1
        
        # Sort by last message time
        conv_list = sorted(conversations.values(), key=lambda x: x['last_message_at'], reverse=True)
        
        return jsonify(create_response(True, data={
            'conversations': conv_list,
            'count': len(conv_list)
        }))
    except Exception as e:
        logger.error(f"Failed to get conversations: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500


@api_bp.route('/farmlink/conversations/<conversation_id>/messages', methods=['GET'])
def get_farmlink_conversation_messages(conversation_id):
    """Get all messages in a conversation."""
    try:
        # For demo, user_id = 1
        user_id = 1
        
        # Get all messages for this conversation
        messages = [m for m in farmlink_messages if m['conversation_id'] == conversation_id]
        
        # Sort by created_at
        messages = sorted(messages, key=lambda x: x['created_at'])
        
        # Add is_mine flag and mark as read
        for msg in messages:
            msg['is_mine'] = msg['sender_id'] == user_id
            msg['sender_name'] = f"User {msg['sender_id']}"
            # Mark as read if current user is receiver
            if msg['receiver_id'] == user_id:
                msg['is_read'] = True
        
        return jsonify(create_response(True, data={
            'messages': messages,
            'count': len(messages)
        }))
    except Exception as e:
        logger.error(f"Failed to get messages: {str(e)}")
        return jsonify(create_response(False, message=str(e))), 500
