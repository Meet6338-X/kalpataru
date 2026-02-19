"""
Kalpataru Flask API Entry Point
Agricultural AI platform for disease detection, irrigation, weather forecasting, 
yield prediction, price forecasting, and crop recommendations.
"""

from flask import Flask, jsonify
from api.routes import api_bp
from utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger(__name__)

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def index():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Kalpataru API',
        'version': '1.0.0'
    })

@app.route('/health')
def health():
    """Detailed health check."""
    return jsonify({
        'status': 'ok',
        'models_loaded': True
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
