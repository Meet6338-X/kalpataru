"""
Mandi Price API Endpoints

REST API endpoints for accessing Government of India Mandi Price data
"""

from flask import Blueprint, request, jsonify
from services.mandi_service import MandiService
import logging

logger = logging.getLogger(__name__)

mandi_bp = Blueprint('mandi', __name__, url_prefix='/api/mandi')


@mandi_bp.route('/prices', methods=['GET'])
def get_prices():
    """Get current daily prices from various markets"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        market = request.args.get('market')
        commodity = request.args.get('commodity')
        variety = request.args.get('variety')
        offset = request.args.get('offset', 0, type=int)
        limit = min(request.args.get('limit', 50, type=int), 500)
        
        result = MandiService.get_daily_prices(
            state=state,
            district=district,
            market=market,
            commodity=commodity,
            variety=variety,
            offset=offset,
            limit=limit
        )
        
        # If external API fails, return fallback data
        if result.get('status') == 'error' or not result.get('records'):
            result = get_fallback_prices(state, district, market, commodity, limit, offset)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_prices: {str(e)}")
        # Return fallback data on error
        return jsonify(get_fallback_prices(
            request.args.get('state'),
            request.args.get('district'),
            request.args.get('market'),
            request.args.get('commodity'),
            min(request.args.get('limit', 50, type=int), 500),
            request.args.get('offset', 0, type=int)
        ))


def get_fallback_prices(state=None, district=None, market=None, commodity=None, limit=50, offset=0):
    """Generate fallback price data when external API is unavailable"""
    import datetime
    
    # Fallback sample data
    fallback_records = [
        {"state": "Maharashtra", "district": "Pune", "market": "Pune APMC", "commodity": "Tomato", "variety": "Local", "min_price": 800, "max_price": 1500, "modal_price": 1200, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Pune", "market": "Pune APMC", "commodity": "Tomato", "variety": "Hybrid", "min_price": 1000, "max_price": 1800, "modal_price": 1400, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Mumbai", "market": "Vashi APMC", "commodity": "Tomato", "variety": "Local", "min_price": 900, "max_price": 1600, "modal_price": 1250, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Nashik", "market": "Nashik APMC", "commodity": "Onion", "variety": "Red", "min_price": 600, "max_price": 1200, "modal_price": 900, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Nagpur", "market": "Nagpur APMC", "commodity": "Wheat", "variety": "Local", "min_price": 1800, "max_price": 2200, "modal_price": 2000, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Ahmednagar", "market": "Ahmednagar APMC", "commodity": "Potato", "variety": "Local", "min_price": 400, "max_price": 800, "modal_price": 600, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Solapur", "market": "Solapur APMC", "commodity": "Sugarcane", "variety": "Local", "min_price": 2500, "max_price": 3500, "modal_price": 3000, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Kolhapur", "market": "Kolhapur APMC", "commodity": "Rice", "variety": "Basmati", "min_price": 4000, "max_price": 6000, "modal_price": 5000, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Sangli", "market": "Sangli APMC", "commodity": "Grapes", "variety": "Thompson", "min_price": 3000, "max_price": 5000, "modal_price": 4000, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Maharashtra", "district": "Aurangabad", "market": "Aurangabad APMC", "commodity": "Cotton", "variety": "Desi", "min_price": 5000, "max_price": 6500, "modal_price": 5800, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Uttar Pradesh", "district": "Lucknow", "market": "Lucknow APMC", "commodity": "Potato", "variety": "Desi", "min_price": 350, "max_price": 700, "modal_price": 500, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Uttar Pradesh", "district": "Agra", "market": "Agra APMC", "commodity": "Tomato", "variety": "Local", "min_price": 700, "max_price": 1300, "modal_price": 1000, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Punjab", "district": "Ludhiana", "market": "Ludhiana APMC", "commodity": "Wheat", "variety": "Sharbati", "min_price": 2000, "max_price": 2400, "modal_price": 2200, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Punjab", "district": "Amritsar", "market": "Amritsar APMC", "commodity": "Rice", "variety": "Pusa Basmati", "min_price": 3500, "max_price": 4500, "modal_price": 4000, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
        {"state": "Gujarat", "district": "Ahmedabad", "market": "Ahmedabad APMC", "commodity": "Cotton", "variety": "BT Cotton", "min_price": 5500, "max_price": 7000, "modal_price": 6200, "price_unit": "Quintal", "arrival_date": "20/02/2026"},
    ]
    
    # Apply filters
    filtered = fallback_records
    if state:
        filtered = [r for r in filtered if state.lower() in r['state'].lower()]
    if district:
        filtered = [r for r in filtered if district.lower() in r['district'].lower()]
    if market:
        filtered = [r for r in filtered if market.lower() in r['market'].lower()]
    if commodity:
        filtered = [r for r in filtered if commodity.lower() in r['commodity'].lower()]
    
    total = len(filtered)
    paginated = filtered[offset:offset + limit]
    
    return {
        'status': 'success',
        'fetched_at': datetime.datetime.now().isoformat(),
        'limit': limit,
        'offset': offset,
        'total_records': total,
        'records': paginated,
        'source': 'fallback'
    }


@mandi_bp.route('/prices/<path:commodity_name>', methods=['GET'])
def get_commodity_prices(commodity_name):
    """Get prices for a specific commodity"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        market = request.args.get('market')
        limit = min(request.args.get('limit', 50, type=int), 200)
        
        result = MandiService.get_daily_prices(
            commodity=commodity_name,
            state=state,
            district=district,
            market=market,
            limit=limit
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_commodity_prices: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch commodity prices'
        }), 500


@mandi_bp.route('/states', methods=['GET'])
def get_states():
    """Get list of all available states"""
    try:
        states = MandiService.get_states()
        return jsonify({
            'status': 'success',
            'count': len(states),
            'states': states
        })
    except Exception as e:
        logger.error(f"Error in get_states: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch states'
        }), 500


@mandi_bp.route('/districts', methods=['GET'])
def get_districts():
    """Get list of districts for a given state"""
    try:
        state = request.args.get('state')
        
        if not state:
            return jsonify({
                'status': 'error',
                'message': 'State parameter is required'
            }), 400
        
        districts = MandiService.get_districts(state)
        return jsonify({
            'status': 'success',
            'state': state,
            'count': len(districts),
            'districts': districts
        })
    except Exception as e:
        logger.error(f"Error in get_districts: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch districts'
        }), 500


@mandi_bp.route('/markets', methods=['GET'])
def get_markets():
    """Get list of markets for a given state and district"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        if not state or not district:
            return jsonify({
                'status': 'error',
                'message': 'Both state and district parameters are required'
            }), 400
        
        markets = MandiService.get_markets(state, district)
        return jsonify({
            'status': 'success',
            'state': state,
            'district': district,
            'count': len(markets),
            'markets': markets
        })
    except Exception as e:
        logger.error(f"Error in get_markets: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch markets'
        }), 500


@mandi_bp.route('/commodities', methods=['GET'])
def get_commodities():
    """Get list of all available commodities"""
    try:
        commodities = MandiService.get_commodities()
        return jsonify({
            'status': 'success',
            'count': len(commodities),
            'commodities': commodities
        })
    except Exception as e:
        logger.error(f"Error in get_commodities: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch commodities'
        }), 500


@mandi_bp.route('/trends/<path:commodity_name>', methods=['GET'])
def get_price_trends(commodity_name):
    """Get price trend analysis for a commodity"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        result = MandiService.get_price_trends(
            commodity=commodity_name,
            state=state,
            district=district
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_price_trends: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to analyze price trends'
        }), 500


@mandi_bp.route('/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard summary data"""
    return jsonify(MandiService.get_dashboard_data())


# ==================== MANDISENSE ADVANCED API ENDPOINTS ====================

@mandi_bp.route('/forecast/<path:commodity_name>', methods=['GET'])
def get_price_forecast(commodity_name):
    """
    Get price forecast for a commodity
    Query params: state, district, market, horizon (7, 14, 30)
    """
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        market = request.args.get('market')
        horizon = request.args.get('horizon', 7, type=int)
        
        # Validate horizon
        if horizon not in [7, 14, 30]:
            horizon = 7
        
        result = MandiService.predict_prices(
            commodity=commodity_name,
            state=state,
            district=district,
            market=market,
            horizon=horizon
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_price_forecast: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate price forecast'
        }), 500


@mandi_bp.route('/demand/<path:commodity_name>', methods=['GET'])
def get_demand_analytics(commodity_name):
    """
    Get demand analytics for a commodity
    Query params: state
    """
    try:
        state = request.args.get('state')
        
        result = MandiService.get_demand_analytics(
            commodity=commodity_name,
            state=state
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_demand_analytics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to analyze demand'
        }), 500


@mandi_bp.route('/recommend', methods=['POST'])
def get_best_mandi_recommendation():
    """
    Get best mandi recommendations for selling
    Body params: commodity, quantity_tons, origin_state, origin_district, transport_cost_per_km
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body is required'
            }), 400
        
        commodity = data.get('commodity')
        quantity_tons = data.get('quantity_tons')
        origin_state = data.get('origin_state')
        origin_district = data.get('origin_district')
        transport_cost_per_km = data.get('transport_cost_per_km', 15)
        
        if not commodity or not quantity_tons or not origin_state:
            return jsonify({
                'status': 'error',
                'message': 'commodity, quantity_tons, and origin_state are required'
            }), 400
        
        result = MandiService.recommend_best_mandi(
            commodity=commodity,
            quantity_tons=float(quantity_tons),
            origin_state=origin_state,
            origin_district=origin_district,
            transport_cost_per_km=float(transport_cost_per_km)
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_best_mandi_recommendation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate recommendations'
        }), 500


@mandi_bp.route('/alerts/check', methods=['GET'])
def check_price_alerts():
    """
    Check if target price is met in any market
    Query params: commodity, target_price, state
    """
    try:
        commodity = request.args.get('commodity')
        target_price = request.args.get('target_price', type=float)
        state = request.args.get('state')
        
        if not commodity or not target_price:
            return jsonify({
                'status': 'error',
                'message': 'commodity and target_price are required'
            }), 400
        
        result = MandiService.check_price_alerts(
            commodity=commodity,
            target_price=target_price,
            state=state
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in check_price_alerts: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to check price alerts'
        }), 500


# ==================== FARMLINK API ENDPOINTS ====================

@mandi_bp.route('/farmlink/search', methods=['GET'])
def search_farmlink_listings():
    """
    Search FarmLink listings
    Query params: commodity, state, min_quantity, max_price
    """
    try:
        commodity = request.args.get('commodity')
        state = request.args.get('state')
        min_quantity = request.args.get('min_quantity', type=float)
        max_price = request.args.get('max_price', type=float)
        
        result = MandiService.search_listings(
            commodity=commodity,
            state=state,
            min_quantity=min_quantity,
            max_price=max_price
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in search_farmlink_listings: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to search listings'
        }), 500


@mandi_bp.route('/farmlink/buyer-forecast/<path:commodity_name>', methods=['GET'])
def get_buyer_demand_forecast(commodity_name):
    """
    Get buyer demand forecast for procurement planning
    Query params: months (1-6)
    """
    try:
        months = request.args.get('months', 3, type=int)
        months = min(max(months, 1), 6)
        
        result = MandiService.get_buyer_demand_forecast(
            commodity=commodity_name,
            months=months
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_buyer_demand_forecast: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate demand forecast'
        }), 500
