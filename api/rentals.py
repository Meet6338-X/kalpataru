"""
Rental Equipment API
Provides endpoints for renting farm equipment, seeds, fertilizers, pesticides, and labor.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import random

# Create Blueprint
rentals_bp = Blueprint('rentals', __name__, url_prefix='/api/rentals')

# Sample data for demonstration - in production, this would come from a database
SAMPLE_EQUIPMENT = [
    {
        'id': 1,
        'equipment_type': 'Tractor',
        'category': 'tractors',
        'location': 'Pune',
        'price_per_day': 2500,
        'rental_type': 'daily',
        'owner_name': 'Gurpreet Singh',
        'available': True
    },
    {
        'id': 2,
        'equipment_type': 'Harvester',
        'category': 'tractors',
        'location': 'Pune',
        'price_per_day': 5000,
        'rental_type': 'daily',
        'owner_name': 'Raj Kumar',
        'available': True
    },
    {
        'id': 3,
        'equipment_type': 'Agricultural Drone',
        'category': 'drones',
        'location': 'Pune',
        'price_per_day': 2500,
        'rental_type': 'daily',
        'owner_name': 'Prakash Gupta',
        'available': True
    },
    {
        'id': 4,
        'equipment_type': 'Spraying Drone',
        'category': 'drones',
        'location': 'Pune',
        'price_per_day': 2000,
        'rental_type': 'daily',
        'owner_name': 'Ramesh Patel',
        'available': True
    },
    {
        'id': 5,
        'equipment_type': 'Rice Seeds',
        'category': 'seeds',
        'location': 'Pune',
        'price': 800,
        'rental_type': 'per_quintal',
        'owner_name': 'Krishna Traders',
        'available': True
    },
    {
        'id': 6,
        'equipment_type': 'Wheat Seeds',
        'category': 'seeds',
        'location': 'Pune',
        'price': 650,
        'rental_type': 'per_quintal',
        'owner_name': 'Amar Seeds Store',
        'available': True
    },
    {
        'id': 7,
        'equipment_type': 'Urea',
        'category': 'fertilizers',
        'location': 'Pune',
        'price': 300,
        'rental_type': 'per_bag',
        'owner_name': 'Agro Chem Store',
        'available': True
    },
    {
        'id': 8,
        'equipment_type': 'DAP',
        'category': 'fertilizers',
        'location': 'Pune',
        'price': 1350,
        'rental_type': 'per_bag',
        'owner_name': 'Farm Solutions',
        'available': True
    },
    {
        'id': 9,
        'equipment_type': 'Insecticide',
        'category': 'pesticides',
        'location': 'Pune',
        'price': 450,
        'rental_type': 'per_liter',
        'owner_name': 'Crop Care India',
        'available': True
    },
    {
        'id': 10,
        'equipment_type': 'Fungicide',
        'category': 'pesticides',
        'location': 'Pune',
        'price': 380,
        'rental_type': 'per_liter',
        'owner_name': 'Agri Products Co',
        'available': True
    },
    {
        'id': 11,
        'equipment_type': 'Farm Worker',
        'category': 'labor',
        'location': 'Pune',
        'price_per_day': 400,
        'rental_type': 'daily',
        'owner_name': 'Labor Union',
        'available': True
    },
    {
        'id': 12,
        'equipment_type': 'Skilled Labor',
        'category': 'labor',
        'location': 'Pune',
        'price_per_day': 600,
        'rental_type': 'daily',
        'owner_name': 'Skilled Workers Assoc',
        'available': True
    },
    {
        'id': 13,
        'equipment_type': 'Machine Operator',
        'category': 'labor',
        'location': 'Pune',
        'price_per_day': 800,
        'rental_type': 'daily',
        'owner_name': 'Tech Farm Services',
        'available': True
    },
    {
        'id': 14,
        'equipment_type': 'Seed Drill',
        'category': 'tractors',
        'location': 'Pune',
        'price_per_day': 1200,
        'rental_type': 'daily',
        'owner_name': 'Mountain Farms',
        'available': True
    },
    {
        'id': 15,
        'equipment_type': 'Sprayer',
        'category': 'tractors',
        'location': 'Pune',
        'price_per_day': 800,
        'rental_type': 'daily',
        'owner_name': 'Coastal Agro',
        'available': True
    },
    {
        'id': 16,
        'equipment_type': 'Tractor',
        'category': 'tractors',
        'location': 'Maharashtra',
        'price_per_day': 2200,
        'rental_type': 'daily',
        'owner_name': 'Maharashtra Farms',
        'available': True
    },
    {
        'id': 17,
        'equipment_type': 'Agricultural Drone',
        'category': 'drones',
        'location': 'Maharashtra',
        'price_per_day': 3200,
        'rental_type': 'daily',
        'owner_name': 'Drone Tech Maharashtra',
        'available': True
    },
    {
        'id': 18,
        'equipment_type': 'Maize Seeds',
        'category': 'seeds',
        'location': 'Maharashtra',
        'price': 700,
        'rental_type': 'per_quintal',
        'owner_name': 'Maharashtra Seeds Ltd',
        'available': True
    },
    {
        'id': 19,
        'equipment_type': 'NPK',
        'category': 'fertilizers',
        'location': 'Maharashtra',
        'price': 1200,
        'rental_type': 'per_bag',
        'owner_name': 'Agro Input Maharashtra',
        'available': True
    },
    {
        'id': 20,
        'equipment_type': 'Herbicide',
        'category': 'pesticides',
        'location': 'Maharashtra',
        'price': 400,
        'rental_type': 'per_liter',
        'owner_name': 'Pest Control Maharashtra',
        'available': True
    }
]


@rentals_bp.route('/equipment', methods=['GET'])
def get_equipment():
    """Get available equipment for rent with optional filters"""
    # Get query parameters
    category = request.args.get('category', '')
    equipment_type = request.args.get('equipment_type', '')
    location = request.args.get('location', '')
    rental_type = request.args.get('rental_type', '')
    max_price = request.args.get('max_price', type=float)
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Filter equipment
    filtered = SAMPLE_EQUIPMENT
    
    if category:
        filtered = [e for e in filtered if e.get('category', '').lower() == category.lower()]
    
    if equipment_type:
        filtered = [e for e in filtered if equipment_type.lower() in e.get('equipment_type', '').lower()]
    
    if location:
        filtered = [e for e in filtered if location.lower() in e.get('location', '').lower()]
    
    if rental_type:
        filtered = [e for e in filtered if e.get('rental_type', '').lower() == rental_type.lower()]
    
    if max_price:
        filtered = [e for e in filtered 
                   if (e.get('price_per_day', 0) <= max_price or e.get('price', 0) <= max_price)]
    
    # Apply pagination
    total = len(filtered)
    paginated = filtered[offset:offset + limit]
    
    return jsonify({
        'success': True,
        'data': {
            'records': paginated,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    })


@rentals_bp.route('/equipment/<int:equipment_id>', methods=['GET'])
def get_equipment_by_id(equipment_id):
    """Get specific equipment by ID"""
    equipment = next((e for e in SAMPLE_EQUIPMENT if e['id'] == equipment_id), None)
    
    if not equipment:
        return jsonify({
            'success': False,
            'message': 'Equipment not found'
        }), 404
    
    return jsonify({
        'success': True,
        'data': equipment
    })


@rentals_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get available rental categories"""
    categories = [
        {'id': 'tractors', 'name': 'Tractors & Machinery', 'icon': '🚜'},
        {'id': 'drones', 'name': 'Drones', 'icon': '🛸'},
        {'id': 'seeds', 'name': 'Seeds', 'icon': '🌱'},
        {'id': 'fertilizers', 'name': 'Fertilizers', 'icon': '🧪'},
        {'id': 'pesticides', 'name': 'Pesticides', 'icon': '💊'},
        {'id': 'labor', 'name': 'Farm Labor', 'icon': '👨‍🌾'}
    ]
    
    return jsonify({
        'success': True,
        'data': {'categories': categories}
    })


@rentals_bp.route('/equipment-types', methods=['GET'])
def get_equipment_types():
    """Get equipment types for a specific category"""
    category = request.args.get('category', '')
    
    types_by_category = {
        'tractors': ['Tractor', 'Harvester', 'Rotavator', 'Cultivator', 'Seed Drill', 'Thresher', 'Sprayer', 'Irrigation Pump'],
        'drones': ['Agricultural Drone', 'Spraying Drone', 'Survey Drone'],
        'seeds': ['Rice Seeds', 'Wheat Seeds', 'Maize Seeds', 'Vegetable Seeds', 'Cotton Seeds', 'Groundnut Seeds'],
        'fertilizers': ['Urea', 'DAP', 'NPK', 'Vermicompost', 'Organic Fertilizer', 'Potash'],
        'pesticides': ['Insecticide', 'Fungicide', 'Herbicide', 'Pesticide', 'Weedicide'],
        'labor': ['Farm Worker', 'Skilled Labor', 'Machine Operator', 'Harvesting Team']
    }
    
    types = types_by_category.get(category.lower(), [])
    
    return jsonify({
        'success': True,
        'data': {'types': types, 'category': category}
    })


@rentals_bp.route('/locations', methods=['GET'])
def get_locations():
    """Get available locations for equipment"""
    locations = list(set(e['location'] for e in SAMPLE_EQUIPMENT))
    locations.sort()
    
    return jsonify({
        'success': True,
        'data': {'locations': locations}
    })
