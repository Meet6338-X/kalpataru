"""
Training Script for Crop Yield Prediction Model.

This script trains a yield prediction regression model using the preprocessed data.
"""

import pickle
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / 'crop_yield'
MODEL_DIR = BASE_DIR / 'models' / 'yield'
ANALYSIS_DIR = BASE_DIR / 'data' / 'analysis' / 'crop_yield'


def load_processed_data():
    """Load the preprocessed data."""
    logger.info("Loading preprocessed data...")
    
    X_train = np.load(PROCESSED_DIR / 'X_train.npy')
    X_test = np.load(PROCESSED_DIR / 'X_test.npy')
    y_train = np.load(PROCESSED_DIR / 'y_train.npy')
    y_test = np.load(PROCESSED_DIR / 'y_test.npy')
    
    with open(PROCESSED_DIR / 'feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    with open(PROCESSED_DIR / 'label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open(PROCESSED_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Target range: {y_train.min():.2f} - {y_train.max():.2f}")
    
    return X_train, X_test, y_train, y_test, feature_cols, encoders, scaler


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6):
    """Train an XGBoost regressor."""
    logger.info(f"Training XGBoost (n_estimators={n_estimators})...")
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train a Random Forest regressor."""
    logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=100, max_depth=3):
    """Train a Gradient Boosting regressor."""
    logger.info(f"Training Gradient Boosting (n_estimators={n_estimators})...")
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    return model


def train_ridge(X_train, y_train, alpha=1.0):
    """Train a Ridge regression model."""
    logger.info(f"Training Ridge regression (alpha={alpha})...")
    
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    logger.info("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (handle zero values)
    y_test_nonzero = np.where(y_test == 0, 1e-10, y_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test_nonzero)) * 100
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }
    
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    return metrics, y_pred


def get_feature_importance(model, feature_cols):
    """Get feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_cols, model.feature_importances_))
        # Convert numpy types to Python native types for JSON serialization
        sorted_importance = {k: float(v) for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)}
        return sorted_importance
    elif hasattr(model, 'coef_'):
        importance = dict(zip(feature_cols, np.abs(model.coef_)))
        # Convert numpy types to Python native types for JSON serialization
        sorted_importance = {k: float(v) for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)}
        return sorted_importance
    return None


def save_model(model, encoders, scaler, feature_cols, metrics, model_name='crop_yield'):
    """Save the trained model and associated files."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = MODEL_DIR / f'{model_name}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")
    
    # Save as XGBoost native format if applicable
    if isinstance(model, xgb.XGBRegressor):
        xgb_path = MODEL_DIR / f'{model_name}_xgboost.json'
        model.save_model(str(xgb_path))
        logger.info(f"Saved XGBoost model to {xgb_path}")
    
    # Save encoders
    encoders_path = MODEL_DIR / f'{model_name}_encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    logger.info(f"Saved encoders to {encoders_path}")
    
    # Save scaler
    scaler_path = MODEL_DIR / f'{model_name}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save feature columns
    features_path = MODEL_DIR / f'{model_name}_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Saved feature columns to {features_path}")
    
    # Save metrics
    metrics_path = ANALYSIS_DIR / f'{model_name}_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save feature importance
    importance = get_feature_importance(model, feature_cols)
    if importance:
        importance_path = ANALYSIS_DIR / f'{model_name}_feature_importance_{timestamp}.json'
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        logger.info(f"Saved feature importance to {importance_path}")


def compare_models(X_train, X_test, y_train, y_test):
    """Compare different models and return the best one."""
    logger.info("Comparing different models...")
    
    results = {}
    
    # XGBoost
    xgb_model = train_xgboost(X_train, y_train, n_estimators=100)
    xgb_metrics, _ = evaluate_model(xgb_model, X_test, y_test)
    results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train, n_estimators=100)
    rf_metrics, _ = evaluate_model(rf_model, X_test, y_test)
    results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train, n_estimators=100)
    gb_metrics, _ = evaluate_model(gb_model, X_test, y_test)
    results['Gradient Boosting'] = {'model': gb_model, 'metrics': gb_metrics}
    
    # Ridge Regression
    ridge_model = train_ridge(X_train, y_train)
    ridge_metrics, _ = evaluate_model(ridge_model, X_test, y_test)
    results['Ridge'] = {'model': ridge_model, 'metrics': ridge_metrics}
    
    # Find best model based on R²
    best_model_name = max(results, key=lambda x: results[x]['metrics']['r2'])
    best_result = results[best_model_name]
    
    logger.info("\nModel Comparison Results:")
    for name, result in results.items():
        logger.info(f"  {name}: R² = {result['metrics']['r2']:.4f}, RMSE = {result['metrics']['rmse']:.4f}")
    
    logger.info(f"\nBest model: {best_model_name}")
    
    return best_result['model'], best_model_name, results


def analyze_predictions(y_test, y_pred, encoders=None):
    """Analyze prediction errors."""
    errors = y_test - y_pred
    
    analysis = {
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
        'error_min': float(np.min(errors)),
        'error_max': float(np.max(errors)),
        'overestimation_count': int(np.sum(errors < 0)),
        'underestimation_count': int(np.sum(errors > 0)),
        'within_10_percent': float(np.mean(np.abs(errors / y_test) < 0.1) * 100),
        'within_20_percent': float(np.mean(np.abs(errors / y_test) < 0.2) * 100)
    }
    
    logger.info("\nPrediction Analysis:")
    logger.info(f"  Mean error: {analysis['error_mean']:.4f}")
    logger.info(f"  Error std: {analysis['error_std']:.4f}")
    logger.info(f"  Predictions within 10%: {analysis['within_10_percent']:.1f}%")
    logger.info(f"  Predictions within 20%: {analysis['within_20_percent']:.1f}%")
    
    return analysis


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Crop Yield Model Training")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n[Step 1/4] Loading preprocessed data...")
    X_train, X_test, y_train, y_test, feature_cols, encoders, scaler = load_processed_data()
    
    # Step 2: Compare models
    logger.info("\n[Step 2/4] Comparing models...")
    best_model, best_model_name, all_results = compare_models(
        X_train, X_test, y_train, y_test
    )
    
    # Step 3: Evaluate best model
    logger.info("\n[Step 3/4] Evaluating best model...")
    metrics, y_pred = evaluate_model(best_model, X_test, y_test)
    
    # Analyze predictions
    analysis = analyze_predictions(y_test, y_pred)
    metrics['analysis'] = analysis
    
    # Step 4: Save model
    logger.info("\n[Step 4/4] Saving model...")
    save_model(best_model, encoders, scaler, feature_cols, metrics)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"Test R²: {metrics['r2']:.4f}")
    logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
    logger.info(f"\nModel saved to: {MODEL_DIR}")


if __name__ == '__main__':
    main()