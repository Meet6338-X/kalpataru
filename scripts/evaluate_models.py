"""
Model Evaluation Script for Kalpataru.

This script provides comprehensive evaluation of trained models.
"""

import pickle
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
ANALYSIS_DIR = BASE_DIR / 'data' / 'analysis'


def evaluate_crop_recommendation():
    """Evaluate the crop recommendation model."""
    logger.info("Evaluating Crop Recommendation Model...")
    
    model_path = MODEL_DIR / 'crop' / 'crop_recommendation_model.pkl'
    encoders_path = MODEL_DIR / 'crop' / 'crop_recommendation_encoders.pkl'
    
    if not model_path.exists():
        logger.warning("Crop recommendation model not found. Skipping.")
        return None
    
    # Load model and encoders
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    # Load test data
    X_test = np.load(PROCESSED_DIR / 'crop_recommendation' / 'X_test.npy')
    y_test = np.load(PROCESSED_DIR / 'crop_recommendation' / 'y_test.npy')
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Classification report
    class_names = encoders['label'].classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save results
    results = {
        'model': 'Crop Recommendation',
        'metrics': metrics,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Save evaluation results
    eval_dir = ANALYSIS_DIR / 'crop_recommendation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(eval_dir / f'evaluation_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, eval_dir / f'confusion_matrix_{timestamp}.png')
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return results


def evaluate_crop_yield():
    """Evaluate the crop yield model."""
    logger.info("Evaluating Crop Yield Model...")
    
    model_path = MODEL_DIR / 'yield' / 'crop_yield_model.pkl'
    
    if not model_path.exists():
        logger.warning("Crop yield model not found. Skipping.")
        return None
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    X_test = np.load(PROCESSED_DIR / 'crop_yield' / 'X_test.npy')
    y_test = np.load(PROCESSED_DIR / 'crop_yield' / 'y_test.npy')
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'mape': float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)
    }
    
    # Prediction analysis
    errors = y_test - y_pred
    analysis = {
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
        'within_10_percent': float(np.mean(np.abs(errors / (y_test + 1e-10)) < 0.1) * 100),
        'within_20_percent': float(np.mean(np.abs(errors / (y_test + 1e-10)) < 0.2) * 100)
    }
    
    results = {
        'model': 'Crop Yield',
        'metrics': metrics,
        'analysis': analysis
    }
    
    # Save evaluation results
    eval_dir = ANALYSIS_DIR / 'crop_yield'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(eval_dir / f'evaluation_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred, eval_dir / f'actual_vs_predicted_{timestamp}.png')
    
    logger.info(f"R² Score: {metrics['r2']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    
    return results


def evaluate_disease_detection():
    """Evaluate the disease detection model."""
    logger.info("Evaluating Disease Detection Model...")
    
    model_path = MODEL_DIR / 'disease' / 'disease_mobilenet_model.h5'
    
    if not model_path.exists():
        logger.warning("Disease detection model not found. Skipping.")
        return None
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except ImportError:
        logger.warning("TensorFlow not available. Skipping disease model evaluation.")
        return None
    
    # Load model
    model = load_model(str(model_path))
    
    # Load class indices
    class_indices_path = MODEL_DIR / 'disease' / 'disease_detection_class_indices.pkl'
    with open(class_indices_path, 'rb') as f:
        class_indices = pickle.load(f)
    
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(class_indices))]
    
    # Load test data using ImageDataGenerator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        PROCESSED_DIR / 'disease_images' / 'test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    
    # Predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'model': 'Disease Detection',
        'metrics': metrics,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Save evaluation results
    eval_dir = ANALYSIS_DIR / 'disease_detection'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(eval_dir / f'evaluation_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot confusion matrix (top 20 classes for readability)
    plot_confusion_matrix(cm[:20, :20], class_names[:20], eval_dir / f'confusion_matrix_{timestamp}.png')
    
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return results


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_actual_vs_predicted(y_true, y_pred, output_path):
    """Plot actual vs predicted values for regression."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved actual vs predicted plot to {output_path}")


def generate_summary_report(results):
    """Generate a summary report of all model evaluations."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    for model_name, result in results.items():
        if result is None:
            continue
        
        logger.info(f"\n{result['model']}:")
        
        if 'metrics' in result:
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
    
    # Save summary
    summary_path = ANALYSIS_DIR / f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nSummary saved to {summary_path}")


def main():
    """Main evaluation function."""
    logger.info("=" * 60)
    logger.info("Model Evaluation Pipeline")
    logger.info("=" * 60)
    
    results = {}
    
    # Evaluate each model
    results['crop_recommendation'] = evaluate_crop_recommendation()
    results['crop_yield'] = evaluate_crop_yield()
    results['disease_detection'] = evaluate_disease_detection()
    
    # Generate summary
    generate_summary_report(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()