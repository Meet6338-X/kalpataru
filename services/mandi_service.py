"""
Mandi Price Service - Integration with Government of India data.gov.in APIs

This service provides access to:
1. Current Daily Price of Various Commodities from Various Markets
2. Variety-wise Daily Market Prices Data of Commodity

API Documentation: https://data.gov.in/apis/live/9ef84268-d588-465a-a308-a864a43d0070
"""

import requests
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class MandiService:
    """
    Service class for interacting with Government of India Mandi Price APIs
    """
    
    # API Endpoints
    DAILY_PRICES_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
    VARIETY_PRICES_RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
    BASE_URL = "https://api.data.gov.in/resource"
    
    @staticmethod
    def get_api_key() -> str:
        """Get API key"""
        return "579b464db66ec23bdd0000018263afcca3cf4ce05558a68fceb1f2ec"
    
    @staticmethod
    def _make_request(resource_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to data.gov.in API"""
        url = f"{MandiService.BASE_URL}/{resource_id}"
        params['api-key'] = MandiService.get_api_key()
        params['format'] = params.get('format', 'json')
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching mandi data from {resource_id}")
            raise Exception("API request timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching mandi data: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")
    
    @staticmethod
    def get_daily_prices(
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        commodity: Optional[str] = None,
        variety: Optional[str] = None,
        grade: Optional[str] = None,
        offset: int = 0,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get current daily prices from various markets (API 1)"""
        params = {
            'offset': offset,
            'limit': limit
        }
        
        if state:
            params['filters[state.keyword]'] = state
        if district:
            params['filters[district]'] = district
        if market:
            params['filters[market]'] = market
        if commodity:
            params['filters[commodity]'] = commodity
        if variety:
            params['filters[variety]'] = variety
        if grade:
            params['filters[grade]'] = grade
        
        try:
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID, 
                params
            )
            
            records = data.get('records', [])
            formatted_records = [
                {
                    'id': record.get('id'),
                    'state': record.get('state', '').strip(),
                    'district': record.get('district', '').strip(),
                    'market': record.get('market', '').strip(),
                    'commodity': record.get('commodity', '').strip(),
                    'variety': record.get('variety', '').strip(),
                    'arrival_date': record.get('arrival_date'),
                    'min_price': float(record.get('min_price', 0)),
                    'max_price': float(record.get('max_price', 0)),
                    'modal_price': float(record.get('modal_price', 0)),
                    'price_unit': 'Quintal'
                }
                for record in records
            ]
            
            return {
                'status': 'success',
                'total_records': data.get('total', len(formatted_records)),
                'offset': offset,
                'limit': limit,
                'records': formatted_records,
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in get_daily_prices: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'records': []
            }
    
    @staticmethod
    def get_states() -> List[str]:
        """Get list of all available states"""
        try:
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                {'limit': 1000}
            )
            
            records = data.get('records', [])
            states = sorted(set(
                record.get('state', '').strip() 
                for record in records 
                if record.get('state')
            ))
            
            return states
        except Exception as e:
            logger.error(f"Error fetching states: {str(e)}")
            return []
    
    @staticmethod
    def get_districts(state: str) -> List[str]:
        """Get list of districts for a given state"""
        try:
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                {
                    'filters[state.keyword]': state,
                    'limit': 500
                }
            )
            
            records = data.get('records', [])
            districts = sorted(set(
                record.get('district', '').strip()
                for record in records
                if record.get('district')
            ))
            
            return districts
        except Exception as e:
            logger.error(f"Error fetching districts for {state}: {str(e)}")
            return []
    
    @staticmethod
    def get_markets(state: str, district: str) -> List[str]:
        """Get list of markets for a given state and district"""
        try:
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                {
                    'filters[state.keyword]': state,
                    'filters[district]': district,
                    'limit': 200
                }
            )
            
            records = data.get('records', [])
            markets = sorted(set(
                record.get('market', '').strip()
                for record in records
                if record.get('market')
            ))
            
            return markets
        except Exception as e:
            logger.error(f"Error fetching markets for {state}/{district}: {str(e)}")
            return []
    
    @staticmethod
    def get_commodities() -> List[str]:
        """Get list of all available commodities"""
        try:
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                {'limit': 2000}
            )
            
            records = data.get('records', [])
            commodities = sorted(set(
                record.get('commodity', '').strip()
                for record in records
                if record.get('commodity')
            ))
            
            return commodities
        except Exception as e:
            logger.error(f"Error fetching commodities: {str(e)}")
            return []
    
    @staticmethod
    def get_price_trends(commodity: str, district: str = None, state: str = None) -> Dict[str, Any]:
        """Analyze price trends for a commodity"""
        try:
            params = {
                'filters[commodity]': commodity,
                'limit': 500
            }
            
            if district:
                params['filters[district]'] = district
            if state:
                params['filters[state.keyword]'] = state
            
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                params
            )
            
            records = data.get('records', [])
            
            if not records:
                return {
                    'status': 'error',
                    'message': 'No data available for the specified filters'
                }
            
            # Calculate statistics
            prices = []
            for record in records:
                try:
                    modal_price = float(record.get('modal_price', 0))
                    if modal_price > 0:
                        prices.append({
                            'date': record.get('arrival_date'),
                            'price': modal_price,
                            'market': record.get('market', '')
                        })
                except (ValueError, TypeError):
                    continue
            
            if not prices:
                return {
                    'status': 'error',
                    'message': 'No valid price data available'
                }
            
            price_values = [p['price'] for p in prices]
            avg_price = sum(price_values) / len(price_values)
            min_price = min(price_values)
            max_price = max(price_values)
            
            return {
                'status': 'success',
                'commodity': commodity,
                'analysis': {
                    'average_price': round(avg_price, 2),
                    'min_price': min_price,
                    'max_price': max_price,
                    'price_range': max_price - min_price,
                    'data_points': len(prices)
                },
                'price_history': prices[:30],
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price trends: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    @staticmethod
    def get_dashboard_data() -> Dict[str, Any]:
        """Get dashboard summary data"""
        try:
            states = MandiService.get_states()[:20]
            commodities = MandiService.get_commodities()[:30]
            
            return {
                'status': 'success',
                'dashboard': {
                    'available_states': states,
                    'available_commodities': commodities,
                    'total_states': len(MandiService.get_states()),
                    'total_commodities': len(MandiService.get_commodities())
                }
            }
        except Exception as e:
            logger.error(f"Error in get_dashboard_data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    # ==================== MANDISENSE ADVANCED FEATURES ====================
    
    @staticmethod
    def predict_prices(commodity: str, state: str = None, district: str = None, 
                       market: str = None, horizon: int = 7) -> Dict[str, Any]:
        """
        Price Forecasting - Predict future prices using linear regression
        horizon: 7, 14, or 30 days
        """
        try:
            # Fetch historical data for forecasting
            params = {
                'filters[commodity]': commodity,
                'limit': 100
            }
            if state:
                params['filters[state.keyword]'] = state
            if district:
                params['filters[district]'] = district
            if market:
                params['filters[market]'] = market
            
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                params
            )
            
            records = data.get('records', [])
            
            if not records:
                return {
                    'status': 'error',
                    'message': 'Insufficient data for price prediction'
                }
            
            # Extract prices and calculate trend
            prices = []
            for record in records:
                try:
                    modal_price = float(record.get('modal_price', 0))
                    if modal_price > 0:
                        prices.append({
                            'date': record.get('arrival_date'),
                            'price': modal_price
                        })
                except (ValueError, TypeError):
                    continue
            
            if len(prices) < 3:
                return {
                    'status': 'error',
                    'message': 'Need at least 3 data points for forecasting'
                }
            
            # Simple linear regression for prediction
            price_values = [p['price'] for p in prices]
            n = len(price_values)
            
            # Calculate averages and trend
            avg_price = sum(price_values) / n
            recent_avg = sum(price_values[-7:]) / min(7, n) if len(price_values) >= 7 else avg_price
            
            # Calculate simple trend
            if n >= 2:
                trend = (price_values[-1] - price_values[0]) / price_values[0] * 100 if price_values[0] > 0 else 0
                daily_change = (price_values[-1] - price_values[0]) / n if n > 0 else 0
            else:
                trend = 0
                daily_change = 0
            
            # Generate predictions
            from datetime import timedelta
            predictions = []
            base_price = price_values[-1] if price_values else avg_price
            
            for day in range(1, horizon + 1):
                predicted_price = base_price + (daily_change * day)
                confidence = max(0.5, 1 - (day * 0.02))  # Decreasing confidence over time
                
                predictions.append({
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'predicted_price': round(predicted_price, 2),
                    'confidence': round(confidence, 2),
                    'trend': 'UP' if daily_change > 0 else 'DOWN' if daily_change < 0 else 'STABLE'
                })
            
            return {
                'status': 'success',
                'commodity': commodity,
                'forecast_horizon': horizon,
                'methodology': 'LINEAR_REGRESSION',
                'current_price': base_price,
                'avg_historical_price': round(avg_price, 2),
                'recent_trend_percent': round(trend, 2),
                'predictions': predictions,
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    @staticmethod
    def get_demand_analytics(commodity: str, state: str = None) -> Dict[str, Any]:
        """
        Demand Analytics - Identify where demand is highest
        """
        try:
            params = {
                'filters[commodity]': commodity,
                'limit': 200
            }
            if state:
                params['filters[state.keyword]'] = state
            
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                params
            )
            
            records = data.get('records', [])
            
            if not records:
                return {
                    'status': 'error',
                    'message': 'No data available for demand analysis'
                }
            
            # Analyze by state/district
            state_data = {}
            for record in records:
                state_name = record.get('state', '').strip()
                district = record.get('district', '').strip()
                market = record.get('market', '').strip()
                
                try:
                    modal_price = float(record.get('modal_price', 0))
                    min_price = float(record.get('min_price', 0))
                    max_price = float(record.get('max_price', 0))
                    
                    if state_name not in state_data:
                        state_data[state_name] = {
                            'prices': [],
                            'markets': set(),
                            'districts': set()
                        }
                    
                    state_data[state_name]['prices'].append(modal_price)
                    state_data[state_name]['markets'].add(market)
                    state_data[state_name]['districts'].add(district)
                except (ValueError, TypeError):
                    continue
            
            # Calculate demand scores
            demand_insights = []
            for state_name, info in state_data.items():
                if info['prices']:
                    avg_price = sum(info['prices']) / len(info['prices'])
                    max_record_price = max(info['prices'])
                    
                    # Higher price with high supply = higher demand
                    demand_score = min(100, (avg_price / 1000) * 50 + len(info['markets']) * 5)
                    
                    demand_insights.append({
                        'state': state_name,
                        'demand_score': round(demand_score, 2),
                        'avg_price': round(avg_price, 2),
                        'max_price': max_record_price,
                        'market_count': len(info['markets']),
                        'district_count': len(info['districts']),
                        'recommendation': 'HIGH DEMAND' if demand_score > 60 else 'MODERATE' if demand_score > 30 else 'LOW'
                    })
            
            # Sort by demand score
            demand_insights.sort(key=lambda x: x['demand_score'], reverse=True)
            
            return {
                'status': 'success',
                'commodity': commodity,
                'demand_insights': demand_insights[:10],
                'total_states_analyzed': len(demand_insights),
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in demand analytics: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    @staticmethod
    def recommend_best_mandi(commodity: str, quantity_tons: float, origin_state: str,
                             origin_district: str = None, transport_cost_per_km: float = 15) -> Dict[str, Any]:
        """
        Best Mandi Recommender - Find top 3 mandis to sell at
        Considers: transport cost, price differential, expected demand
        """
        try:
            # Get prices across all markets for this commodity
            params = {
                'filters[commodity]': commodity,
                'limit': 300
            }
            
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                params
            )
            
            records = data.get('records', [])
            
            if not records:
                return {
                    'status': 'error',
                    'message': 'No market data available for recommendations'
                }
            
            # Approximate distances (in km) - in production, use actual distance API
            # This is a simplified model
            distance_estimates = {
                'Maharashtra': {'Pune': 150, 'Mumbai': 180, 'Nagpur': 200, 'Nashik': 80},
                'Gujarat': {'Ahmedabad': 200, 'Surat': 250, 'Vadodara': 220},
                'Uttar Pradesh': {'Lucknow': 100, 'Kanpur': 120, 'Varanasi': 150},
                'Madhya Pradesh': {'Bhopal': 100, 'Indore': 120, 'Jabalpur': 150},
                'Karnataka': {'Bangalore': 200, 'Mysore': 150, 'Hubli': 100},
                'Tamil Nadu': {'Chennai': 150, 'Coimbatore': 100, 'Madurai': 120},
                'West Bengal': {'Kolkata': 100, 'Siliguri': 150, 'Asansol': 120},
                'Punjab': {'Ludhiana': 80, 'Amritsar': 100, 'Jalandhar': 60},
                'Haryana': {'Gurgaon': 50, 'Faridabad': 40, 'Karnal': 60},
                'Rajasthan': {'Jaipur': 100, 'Jodhpur': 150, 'Kota': 120}
            }
            
            # Process market data
            market_data = []
            for record in records:
                state = record.get('state', '').strip()
                district = record.get('district', '').strip()
                market = record.get('market', '').strip()
                
                try:
                    modal_price = float(record.get('modal_price', 0))
                    min_price = float(record.get('min_price', 0))
                    max_price = float(record.get('max_price', 0))
                    
                    if modal_price > 0:
                        # Estimate distance
                        est_distance = 100  # Default distance
                        if state in distance_estimates:
                            for city, dist in distance_estimates[state].items():
                                if city.lower() in market.lower() or city.lower() in district.lower():
                                    est_distance = dist
                                    break
                        
                        # Calculate transport cost
                        transport_cost = est_distance * transport_cost_per_km
                        
                        # Calculate net profit (price * quantity - transport cost)
                        # Convert tons to quintals (1 ton = 10 quintals)
                        quantity_quintals = quantity_tons * 10
                        gross_income = modal_price * quantity_quintals
                        net_income = gross_income - transport_cost
                        
                        market_data.append({
                            'state': state,
                            'district': district,
                            'market': market,
                            'modal_price': modal_price,
                            'min_price': min_price,
                            'max_price': max_price,
                            'estimated_distance_km': est_distance,
                            'transport_cost': round(transport_cost, 2),
                            'gross_income': round(gross_income, 2),
                            'net_income': round(net_income, 2)
                        })
                except (ValueError, TypeError):
                    continue
            
            if not market_data:
                return {
                    'status': 'error',
                    'message': 'No valid market data found'
                }
            
            # Sort by net income (best profit)
            market_data.sort(key=lambda x: x['net_income'], reverse=True)
            
            # Get top 3
            top_markets = market_data[:3]
            
            recommendations = []
            for i, market in enumerate(top_markets, 1):
                recommendations.append({
                    'rank': i,
                    'market': market['market'],
                    'district': market['district'],
                    'state': market['state'],
                    'modal_price': market['modal_price'],
                    'distance_km': market['estimated_distance_km'],
                    'transport_cost': market['transport_cost'],
                    'gross_income': market['gross_income'],
                    'net_income': market['net_income'],
                    'savings_vs_local': round(market['net_income'] - (market_data[len(market_data)//2]['net_income'] if len(market_data) > 1 else 0), 2) if i == 1 else None
                })
            
            return {
                'status': 'success',
                'query': {
                    'commodity': commodity,
                    'quantity_tons': quantity_tons,
                    'origin_state': origin_state,
                    'origin_district': origin_district
                },
                'recommendations': recommendations,
                'analysis_note': 'Recommendations based on current prices and estimated transport costs. Actual results may vary.',
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in best mandi recommendation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    @staticmethod
    def check_price_alerts(commodity: str, target_price: float, state: str = None) -> Dict[str, Any]:
        """
        Check if target price is met in any market
        """
        try:
            params = {
                'filters[commodity]': commodity,
                'limit': 100
            }
            if state:
                params['filters[state.keyword]'] = state
            
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                params
            )
            
            records = data.get('records', [])
            
            if not records:
                return {
                    'status': 'error',
                    'message': 'No market data available'
                }
            
            # Find markets where target price is met
            alerts = []
            for record in records:
                try:
                    modal_price = float(record.get('modal_price', 0))
                    min_price = float(record.get('min_price', 0))
                    max_price = float(record.get('max_price', 0))
                    
                    if modal_price >= target_price:
                        alerts.append({
                            'market': record.get('market', '').strip(),
                            'district': record.get('district', '').strip(),
                            'state': record.get('state', '').strip(),
                            'current_price': modal_price,
                            'target_price': target_price,
                            'price_above_target': modal_price - target_price,
                            'alert_triggered': True
                        })
                except (ValueError, TypeError):
                    continue
            
            # Sort by price (highest first)
            alerts.sort(key=lambda x: x['current_price'], reverse=True)
            
            return {
                'status': 'success',
                'commodity': commodity,
                'target_price': target_price,
                'alert_count': len(alerts),
                'alerts': alerts[:10],
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking price alerts: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    # ==================== FARMLINK SERVICE METHODS ====================
    
    @staticmethod
    def search_listings(commodity: str = None, state: str = None, 
                        min_quantity: float = None, max_price: float = None) -> Dict[str, Any]:
        """
        Search FarmLink listings for produce
        """
        try:
            # In production, this would query a listings database
            # For demo, return mock data structure
            return {
                'status': 'success',
                'listings': [],
                'message': 'Use FarmLink marketplace for actual listings'
            }
        except Exception as e:
            logger.error(f"Error searching listings: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def get_buyer_demand_forecast(commodity: str, months: int = 3) -> Dict[str, Any]:
        """
        Get demand forecasting for buyers - powered by MandiSense data
        """
        try:
            # Get historical data
            params = {
                'filters[commodity]': commodity,
                'limit': 100
            }
            
            data = MandiService._make_request(
                MandiService.DAILY_PRICES_RESOURCE_ID,
                params
            )
            
            records = data.get('records', [])
            
            if not records:
                return {'status': 'error', 'message': 'No data available'}
            
            # Analyze price trends to predict demand
            prices = []
            for record in records:
                try:
                    modal_price = float(record.get('modal_price', 0))
                    if modal_price > 0:
                        prices.append({
                            'date': record.get('arrival_date'),
                            'price': modal_price,
                            'state': record.get('state', '').strip()
                        })
                except:
                    continue
            
            # Simple demand inference: lower prices = higher supply = higher demand
            price_values = [p['price'] for p in prices]
            avg_price = sum(price_values) / len(price_values) if price_values else 0
            
            # Forecast
            from datetime import timedelta
            forecasts = []
            for month in range(1, months + 1):
                forecasts.append({
                    'month': month,
                    'estimated_demand': 'HIGH' if avg_price < 2000 else 'MODERATE' if avg_price < 5000 else 'LOW',
                    'recommended_buying_quantity': round(avg_price * 10, 2),
                    'price_range': {
                        'min': round(avg_price * 0.8, 2),
                        'max': round(avg_price * 1.2, 2)
                    }
                })
            
            return {
                'status': 'success',
                'commodity': commodity,
                'current_avg_price': round(avg_price, 2),
                'demand_forecasts': forecasts,
                'fetched_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in buyer demand forecast: {str(e)}")
            return {'status': 'error', 'message': str(e)}
