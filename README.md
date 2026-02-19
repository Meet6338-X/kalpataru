# 🌾 Kalpataru - Agricultural AI Platform

An intelligent agricultural platform that provides disease detection, irrigation optimization, weather forecasting, yield prediction, price forecasting, and crop recommendations using advanced machine learning models.

## Features

- 🦠 **Disease Detection** - CNN-based plant disease detection from leaf images (39 disease classes)
- 💧 **Irrigation Optimization** - Smart irrigation recommendations based on soil and weather conditions
- 🌤️ **Weather Forecasting** - LSTM-based weather prediction for better planning
- 📈 **Yield Prediction** - XGBoost model for crop yield forecasting
- 💰 **Price Forecasting** - Prophet-based commodity price predictions
- 🌱 **Crop Recommendation** - AI-powered crop suggestions based on soil and climate

## Project Structure

```
kalpataru/
├── app.py                  # Flask API Entry Point
├── streamlit_app.py        # Streamlit UI Entry Point
│
├── config/
│   ├── settings.py         # Configuration settings
│   ├── constants.py        # Application constants
│   └── training_config.py  # Training hyperparameters
│
├── data/
│   ├── raw/                # Raw data storage
│   │   ├── crop_recommendation/
│   │   ├── crop_yield/
│   │   └── disease_images/
│   ├── processed/          # Processed data
│   │   ├── crop_recommendation/
│   │   ├── crop_yield/
│   │   └── disease_images/
│   ├── external/           # External data sources
│   └── analysis/           # Analysis notebooks and results
│
├── models/
│   ├── disease/            # CNN disease detection
│   ├── irrigation/         # Irrigation model
│   ├── weather/            # LSTM weather forecasting
│   ├── yield/              # XGBoost yield prediction
│   ├── price/              # Prophet price forecasting
│   └── crop/               # Crop recommendation
│
├── scripts/                # Training and preprocessing scripts
│   ├── organize_data.py
│   ├── preprocess_crop_recommendation.py
│   ├── preprocess_crop_yield.py
│   ├── preprocess_disease_images.py
│   ├── train_crop_recommendation.py
│   ├── train_crop_yield.py
│   ├── train_disease_detection.py
│   ├── evaluate_models.py
│   └── run_pipeline.py
│
├── pipelines/
│   ├── image_pipeline.py
│   ├── weather_pipeline.py
│   ├── price_pipeline.py
│   └── yield_pipeline.py
│
├── services/
│   ├── explainability.py
│   ├── recommendation_engine.py
│   └── translation.py
│
├── api/
│   ├── routes.py          # API endpoints
│   └── schemas.py         # Request/response schemas
│
├── utils/
│   ├── logger.py          # Logging utility
│   └── helpers.py         # Helper functions
│
├── plans/                  # Planning documents
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kalpataru
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Flask API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI will open at `http://localhost:8501`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/disease/predict` | POST | Disease detection |
| `/api/irrigation/predict` | POST | Irrigation prediction |
| `/api/weather/predict` | POST | Weather forecasting |
| `/api/yield/predict` | POST | Yield prediction |
| `/api/price/predict` | POST | Price forecasting |
| `/api/crop/recommend` | POST | Crop recommendation |
| `/api/translate` | POST | Translation service |

## Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Disease Detection | CNN (MobileNetV2) | Plant disease classification (39 classes) |
| Weather Forecasting | LSTM | Time-series weather prediction |
| Yield Prediction | XGBoost | Crop yield forecasting |
| Price Forecasting | Prophet | Commodity price prediction |
| Crop Recommendation | Random Forest/XGBoost | Optimal crop selection |

## Training Pipeline

### Quick Start

Run the complete training pipeline:

```bash
# Run full pipeline (organize data, preprocess, train)
python scripts/run_pipeline.py

# Run only training (skip data organization and preprocessing)
python scripts/run_pipeline.py --train-only

# Run only preprocessing
python scripts/run_pipeline.py --preprocess-only
```

### Individual Scripts

```bash
# Step 1: Organize raw data
python scripts/organize_data.py

# Step 2: Preprocess data
python scripts/preprocess_crop_recommendation.py
python scripts/preprocess_crop_yield.py
python scripts/preprocess_disease_images.py

# Step 3: Train models
python scripts/train_crop_recommendation.py
python scripts/train_crop_yield.py
python scripts/train_disease_detection.py

# Step 4: Evaluate models
python scripts/evaluate_models.py
```

### Dataset Requirements

Place your datasets in the `Dataset/` folder:

```
Dataset/
├── crop recommendations/
│   └── Crop_recommendation.csv
├── crop yield/
│   └── crop_yield.csv
└── Plant_leave_diseases_dataset_with_augmentation/
    ├── Apple___Apple_scab/
    ├── Apple___healthy/
    └── ... (39 disease classes)
```

### Training Configuration

Training parameters can be customized in `config/training_config.py`:

```python
# Example: Crop Recommendation settings
CROP_CONFIG = {
    'model_type': 'random_forest',
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
    }
}

# Example: Disease Detection settings
DISEASE_CONFIG = {
    'model_type': 'transfer_learning',
    'base_model': 'mobilenet',
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 50
}
```

## Supported Languages

- English (en)
- Hindi (hi)
- Telugu (te)
- Tamil (ta)
- Marathi (mr)
- Bengali (bn)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)

## Configuration

Environment variables can be set in `.env` file:

```
API_HOST=0.0.0.0
API_PORT=5000
DEBUG=True
LOG_LEVEL=INFO
```

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black .
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
