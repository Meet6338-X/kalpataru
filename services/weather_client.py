"""
Weather API Client Service for Kalpataru.
Integration with OpenWeatherMap and other weather APIs.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random

from utils.logger import setup_logger

logger = setup_logger(__name__)


class WeatherAPIClient:
    """
    Client for interacting with third-party weather APIs.
    Includes retry logic, caching, and mock data for development.
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        base_url: str = "https://api.openweathermap.org/data/2.5"
    ):
        """
        Initialize the weather API client.
        
        Args:
            api_key: API key for weather service
            base_url: Base URL for the weather API
        """
        self.api_key = api_key or os.getenv('WEATHER_API_KEY', 'MOCK_API_KEY')
        self.base_url = base_url
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour cache
    
    def get_current_weather(
        self, 
        location: str, 
        retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch current weather for a location with automatic retries.
        
        Args:
            location: City name or coordinates
            retries: Number of retry attempts
            
        Returns:
            Weather data dictionary or None if failed
        """
        # Check cache first
        cache_key = f"current_{location}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        for attempt in range(retries):
            try:
                # Mock mode if no real API key
                if self.api_key == 'MOCK_API_KEY':
                    result = self._get_mock_weather(location)
                else:
                    result = self._make_api_request('/weather', params)
                
                if result:
                    self._add_to_cache(cache_key, result)
                    return result
                    
            except Exception as e:
                logger.warning(
                    f"Weather API attempt {attempt + 1} failed for {location}: {str(e)}"
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Weather API finalized failure for {location}")
                    return self._get_mock_weather(location)
        
        return None
    
    def get_forecast(
        self, 
        location: str, 
        days: int = 7,
        retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch weather forecast for a location.
        
        Args:
            location: City name or coordinates
            days: Number of forecast days
            retries: Number of retry attempts
            
        Returns:
            Forecast data dictionary or None if failed
        """
        cache_key = f"forecast_{location}_{days}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8  # 8 readings per day (3-hour intervals)
        }
        
        for attempt in range(retries):
            try:
                if self.api_key == 'MOCK_API_KEY':
                    result = self._get_mock_forecast(location, days)
                else:
                    result = self._make_api_request('/forecast', params)
                
                if result:
                    self._add_to_cache(cache_key, result)
                    return result
                    
            except Exception as e:
                logger.warning(f"Forecast API attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return self._get_mock_forecast(location, days)
        
        return None
    
    def get_agricultural_weather(
        self, 
        location: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get weather data formatted for agricultural use.
        
        Args:
            location: Location name
            days: Number of days for forecast
            
        Returns:
            Agricultural weather summary
        """
        current = self.get_current_weather(location)
        forecast = self.get_forecast(location, days)
        
        if not current:
            current = self._get_mock_weather(location)
        if not forecast:
            forecast = self._get_mock_forecast(location, days)
        
        # Process for agricultural insights
        return self._process_agricultural_weather(current, forecast)
    
    def _make_api_request(
        self, 
        endpoint: str, 
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make actual API request."""
        try:
            import requests
            response = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except ImportError:
            logger.warning("requests library not available")
            return None
        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            raise
    
    def _get_mock_weather(self, location: str) -> Dict[str, Any]:
        """Generate mock weather data for development/testing."""
        # Use location to seed somewhat consistent data
        seed = sum(ord(c) for c in location)
        random.seed(seed)
        
        base_temp = random.uniform(20, 35)
        base_humidity = random.uniform(40, 80)
        
        conditions = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Thunderstorm']
        condition = random.choice(conditions)
        
        return {
            'main': {
                'temp': round(base_temp, 1),
                'feels_like': round(base_temp + random.uniform(-2, 2), 1),
                'temp_min': round(base_temp - 3, 1),
                'temp_max': round(base_temp + 3, 1),
                'pressure': round(random.uniform(1000, 1020), 0),
                'humidity': round(base_humidity, 0)
            },
            'weather': [{
                'main': condition,
                'description': f'{condition.lower()} conditions',
                'icon': '01d' if condition == 'Clear' else '02d'
            }],
            'wind': {
                'speed': round(random.uniform(1, 15), 1),
                'deg': round(random.uniform(0, 360), 0)
            },
            'rain': {
                '1h': round(random.uniform(0, 10), 1)
            } if condition in ['Rain', 'Drizzle', 'Thunderstorm'] else {},
            'clouds': {
                'all': round(random.uniform(0, 100), 0)
            },
            'name': location,
            'dt': int(time.time()),
            'sys': {
                'sunrise': int(time.time() - 21600),  # 6 hours ago
                'sunset': int(time.time() + 21600)    # 6 hours from now
            }
        }
    
    def _get_mock_forecast(self, location: str, days: int) -> Dict[str, Any]:
        """Generate mock forecast data for development/testing."""
        seed = sum(ord(c) for c in location)
        random.seed(seed)
        
        base_temp = random.uniform(20, 35)
        base_humidity = random.uniform(40, 80)
        
        forecast_list = []
        current_time = int(time.time())
        
        conditions = ['Clear', 'Clouds', 'Rain', 'Drizzle']
        
        for i in range(days * 8):  # 8 readings per day
            # Add some variation
            temp_variation = random.uniform(-5, 5)
            humidity_variation = random.uniform(-10, 10)
            
            forecast_list.append({
                'dt': current_time + (i * 10800),  # 3-hour intervals
                'main': {
                    'temp': round(base_temp + temp_variation, 1),
                    'feels_like': round(base_temp + temp_variation + random.uniform(-1, 1), 1),
                    'temp_min': round(base_temp + temp_variation - 2, 1),
                    'temp_max': round(base_temp + temp_variation + 2, 1),
                    'humidity': round(max(20, min(100, base_humidity + humidity_variation)), 0)
                },
                'weather': [{
                    'main': random.choice(conditions),
                    'description': 'weather conditions'
                }],
                'wind': {
                    'speed': round(random.uniform(1, 15), 1)
                },
                'pop': round(random.uniform(0, 1), 2)  # Probability of precipitation
            })
        
        return {
            'list': forecast_list,
            'city': {
                'name': location,
                'country': 'IN'
            }
        }
    
    def _process_agricultural_weather(
        self, 
        current: Dict[str, Any], 
        forecast: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process weather data for agricultural insights."""
        # Extract current conditions
        current_data = {
            'temperature': current.get('main', {}).get('temp', 25),
            'humidity': current.get('main', {}).get('humidity', 60),
            'pressure': current.get('main', {}).get('pressure', 1013),
            'wind_speed': current.get('wind', {}).get('speed', 5),
            'condition': current.get('weather', [{}])[0].get('main', 'Clear'),
            'description': current.get('weather', [{}])[0].get('description', 'clear')
        }
        
        # Process forecast
        forecast_data = []
        rain_days = 0
        total_rain = 0
        temps = []
        humidities = []
        
        forecast_list = forecast.get('list', [])
        
        # Group by day
        daily_data = {}
        for item in forecast_list:
            dt = datetime.fromtimestamp(item['dt'])
            date_str = dt.strftime('%Y-%m-%d')
            
            if date_str not in daily_data:
                daily_data[date_str] = {
                    'temps': [],
                    'humidity': [],
                    'rain': 0,
                    'condition': 'Clear'
                }
            
            daily_data[date_str]['temps'].append(item['main']['temp'])
            daily_data[date_str]['humidity'].append(item['main']['humidity'])
            
            # Check for rain
            if item.get('rain', {}).get('1h', 0) > 0:
                daily_data[date_str]['rain'] += item['rain']['1h']
            
            # Get most common condition
            condition = item['weather'][0]['main']
            if condition in ['Rain', 'Drizzle', 'Thunderstorm']:
                daily_data[date_str]['condition'] = 'Rain'
        
        # Summarize daily data
        for date_str, data in daily_data.items():
            avg_temp = sum(data['temps']) / len(data['temps'])
            avg_humidity = sum(data['humidity']) / len(data['humidity'])
            
            forecast_data.append({
                'date': date_str,
                'temperature_avg': round(avg_temp, 1),
                'temperature_min': round(min(data['temps']), 1),
                'temperature_max': round(max(data['temps']), 1),
                'humidity_avg': round(avg_humidity, 0),
                'condition': data['condition'],
                'rain_mm': round(data['rain'], 1)
            })
            
            temps.extend(data['temps'])
            humidities.extend(data['humidity'])
            if data['rain'] > 0:
                rain_days += 1
                total_rain += data['rain']
        
        # Generate agricultural insights
        insights = self._generate_agricultural_insights(
            current_data, forecast_data, rain_days, total_rain
        )
        
        return {
            'location': current.get('name', 'Unknown'),
            'current': current_data,
            'forecast': forecast_data[:7],  # Limit to 7 days
            'summary': {
                'avg_temperature': round(sum(temps) / len(temps), 1) if temps else 25,
                'avg_humidity': round(sum(humidities) / len(humidities), 0) if humidities else 60,
                'rainy_days': rain_days,
                'total_rain_mm': round(total_rain, 1)
            },
            'agricultural_insights': insights
        }
    
    def _generate_agricultural_insights(
        self,
        current: Dict[str, Any],
        forecast: List[Dict[str, Any]],
        rain_days: int,
        total_rain: float
    ) -> Dict[str, Any]:
        """Generate agricultural insights from weather data."""
        insights = {
            'irrigation_needed': True,
            'spraying_possible': True,
            'harvesting_conditions': 'favorable',
            'alerts': [],
            'recommendations': []
        }
        
        # Check current conditions
        if current['condition'] in ['Rain', 'Drizzle', 'Thunderstorm']:
            insights['spraying_possible'] = False
            insights['alerts'].append('Current rain conditions - avoid spraying')
        
        # Check temperature
        temp = current['temperature']
        if temp > 35:
            insights['alerts'].append('High temperature - consider shade protection')
            insights['recommendations'].append('Irrigate early morning or late evening')
        elif temp < 15:
            insights['alerts'].append('Low temperature - protect sensitive crops')
        
        # Check humidity
        humidity = current['humidity']
        if humidity > 80:
            insights['alerts'].append('High humidity - monitor for fungal diseases')
            insights['recommendations'].append('Ensure good air circulation')
        
        # Check forecast
        if rain_days >= 4:
            insights['irrigation_needed'] = False
            insights['recommendations'].append('Reduce irrigation - rain expected')
        
        if total_rain > 50:
            insights['alerts'].append('Heavy rainfall expected - ensure drainage')
        
        # Check for consecutive dry days
        dry_days = len([d for d in forecast if d['condition'] not in ['Rain', 'Drizzle']])
        if dry_days >= 5:
            insights['irrigation_needed'] = True
            insights['recommendations'].append('Plan irrigation for dry spell')
        
        return insights
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
        return None
    
    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        """Add data to cache."""
        self._cache[key] = (data, time.time())


def get_weather(location: str, days: int = 7) -> Dict[str, Any]:
    """
    Main function to get weather data.
    
    Args:
        location: Location name
        days: Number of forecast days
        
    Returns:
        Weather data dictionary
    """
    client = WeatherAPIClient()
    return client.get_agricultural_weather(location, days)


def get_current_weather(location: str) -> Dict[str, Any]:
    """
    Get current weather for a location.
    
    Args:
        location: Location name
        
    Returns:
        Current weather data
    """
    client = WeatherAPIClient()
    return client.get_current_weather(location) or {}


def get_weather_forecast(location: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Get weather forecast for a location.
    
    Args:
        location: Location name
        days: Number of forecast days
        
    Returns:
        List of daily forecast data
    """
    client = WeatherAPIClient()
    forecast = client.get_forecast(location, days)
    
    if not forecast:
        return []
    
    # Process into daily summaries
    daily_data = {}
    for item in forecast.get('list', []):
        dt = datetime.fromtimestamp(item['dt'])
        date_str = dt.strftime('%Y-%m-%d')
        
        if date_str not in daily_data:
            daily_data[date_str] = {
                'date': date_str,
                'temps': [],
                'humidity': [],
                'conditions': []
            }
        
        daily_data[date_str]['temps'].append(item['main']['temp'])
        daily_data[date_str]['humidity'].append(item['main']['humidity'])
        daily_data[date_str]['conditions'].append(item['weather'][0]['main'])
    
    result = []
    for date_str, data in daily_data.items():
        from collections import Counter
        condition_counts = Counter(data['conditions'])
        most_common_condition = condition_counts.most_common(1)[0][0]
        
        result.append({
            'date': date_str,
            'temperature_min': round(min(data['temps']), 1),
            'temperature_max': round(max(data['temps']), 1),
            'temperature_avg': round(sum(data['temps']) / len(data['temps']), 1),
            'humidity_avg': round(sum(data['humidity']) / len(data['humidity']), 0),
            'condition': most_common_condition
        })
    
    return result[:days]
