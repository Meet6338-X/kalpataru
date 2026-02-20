"""
Kalpataru Flask API Entry Point
Agricultural AI platform for disease detection, irrigation, weather forecasting, 
yield prediction, price forecasting, and crop recommendations.
"""

from flask import Flask, jsonify, send_from_directory, send_file
from flask_cors import CORS
from api.routes import api_bp
from utils.logger import setup_logger
import os

# Create Flask app with static folder
app = Flask(__name__, 
            static_folder='frontend',
            static_url_path='')
logger = setup_logger(__name__)

# Enable CORS for API access from frontend
CORS(app)

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def index():
    """Serve the main frontend application."""
    try:
        return send_file('frontend/index.html')
    except:
        return jsonify({
            'status': 'healthy',
            'service': 'Kalpataru API',
            'version': '2.0.0',
            'message': 'Frontend not found. API is running.'
        })

@app.route('/health')
def health():
    """Detailed health check."""
    return jsonify({
        'status': 'ok',
        'models_loaded': True,
        'version': '2.0.0'
    })

# Serve static files
@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files."""
    return send_from_directory('frontend/css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files."""
    return send_from_directory('frontend/js', filename)

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve image files."""
    return send_from_directory('frontend/images', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
