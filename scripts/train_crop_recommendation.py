"""
Training Script for Crop Recommendation Model.

This script trains a crop recommendation classifier using the preprocessed data.
"""

import pickle
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / 'crop_recommendation'
MODEL_DIR = BASE_DIR / 'models' / 'crop'
ANALYSIS_DIR = BASE_DIR / 'data' / 'analysis' / 'crop_recommendation'


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
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Number of classes: {len(encoders['label'].classes_)}")
    
    return X_train, X_test, y_train, y_test, feature_cols, encoders


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train a Random Forest classifier."""
    logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6):
    """Train an XGBoost classifier."""
    logger.info(f"Training XGBoost (n_estimators={n_estimators})...")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=100, max_depth=3):
    """Train a Gradient Boosting classifier."""
    logger.info(f"Training Gradient Boosting (n_estimators={n_estimators})...")
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test, encoders):
    """Evaluate the model and return metrics."""
    logger.info("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Classification report
    class_names = encoders['label'].classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    return metrics, y_pred


def get_feature_importance(model, feature_cols):
    """Get feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_cols, model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return sorted_importance
    return None


def save_model(model, encoders, feature_cols, metrics, model_name='crop_recommendation'):
    """Save the trained model and associated files."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = MODEL_DIR / f'{model_name}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")
    
    # Save encoders with model
    encoders_path = MODEL_DIR / f'{model_name}_encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    logger.info(f"Saved encoders to {encoders_path}")
    
    # Save feature columns
    features_path = MODEL_DIR / f'{model_name}_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Saved feature columns to {features_path}")
    
    # Save metrics
    metrics_path = ANALYSIS_DIR / f'{model_name}_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save feature importance
    importance = get_feature_importance(model, feature_cols)
    if importance:
        importance_path = ANALYSIS_DIR / f'{model_name}_feature_importance_{timestamp}.json'
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        logger.info(f"Saved feature importance to {importance_path}")


def hyperparameter_search(X_train, y_train, model_type='random_forest'):
    """Perform simple hyperparameter search."""
    logger.info(f"Performing hyperparameter search for {model_type}...")
    
    best_score = 0
    best_model = None
    best_params = None
    
    if model_type == 'random_forest':
        param_grid = [
            {'n_estimators': 50, 'max_depth': 10},
            {'n_estimators': 100, 'max_depth': 15},
            {'n_estimators': 100, 'max_depth': None},
            {'n_estimators': 200, 'max_depth': 20},
        ]
        
        for params in param_grid:
            model = train_random_forest(X_train, y_train, **params)
            score = model.score(X_train, y_train)
            logger.info(f"Params: {params}, Train Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
    
    elif model_type == 'xgboost':
        param_grid = [
            {'n_estimators': 50, 'max_depth': 4},
            {'n_estimators': 100, 'max_depth': 6},
            {'n_estimators': 100, 'max_depth': 8},
            {'n_estimators': 200, 'max_depth': 6},
        ]
        
        for params in param_grid:
            model = train_xgboost(X_train, y_train, **params)
            score = model.score(X_train, y_train)
            logger.info(f"Params: {params}, Train Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
    
    logger.info(f"Best params: {best_params}, Best score: {best_score:.4f}")
    
    return best_model, best_params


def compare_models(X_train, X_test, y_train, y_test, encoders):
    """Compare different models and return the best one."""
    logger.info("Comparing different models...")
    
    results = {}
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train, n_estimators=100)
    rf_metrics, _ = evaluate_model(rf_model, X_test, y_test, encoders)
    results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # XGBoost
    xgb_model = train_xgboost(X_train, y_train, n_estimators=100)
    xgb_metrics, _ = evaluate_model(xgb_model, X_test, y_test, encoders)
    results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train, n_estimators=100)
    gb_metrics, _ = evaluate_model(gb_model, X_test, y_test, encoders)
    results['Gradient Boosting'] = {'model': gb_model, 'metrics': gb_metrics}
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['metrics']['accuracy'])
    best_result = results[best_model_name]
    
    logger.info("\nModel Comparison Results:")
    for name, result in results.items():
        logger.info(f"  {name}: Accuracy = {result['metrics']['accuracy']:.4f}")
    
    logger.info(f"\nBest model: {best_model_name}")
    
    return best_result['model'], best_model_name, results


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Crop Recommendation Model Training")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n[Step 1/4] Loading preprocessed data...")
    X_train, X_test, y_train, y_test, feature_cols, encoders = load_processed_data()
    
    # Step 2: Compare models
    logger.info("\n[Step 2/4] Comparing models...")
    best_model, best_model_name, all_results = compare_models(
        X_train, X_test, y_train, y_test, encoders
    )
    
    # Step 3: Evaluate best model
    logger.info("\n[Step 3/4] Evaluating best model...")
    metrics, y_pred = evaluate_model(best_model, X_test, y_test, encoders)
    
    # Step 4: Save model
    logger.info("\n[Step 4/4] Saving model...")
    save_model(best_model, encoders, feature_cols, metrics)
    
    # Print classification report
    logger.info("\n" + "=" * 60)
    logger.info("Classification Report")
    logger.info("=" * 60)
    class_names = encoders['label'].classes_
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"\nModel saved to: {MODEL_DIR}")


if __name__ == '__main__':
    main()