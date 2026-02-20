"""
Training script for Fertilizer Recommendation Model.
Trains a classifier to predict the best fertilizer based on soil and crop parameters.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, f1_score
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from config.training_config import FERTILIZER_CONFIG

logger = setup_logger(__name__)


class FertilizerModelTrainer:
    """Trainer class for fertilizer recommendation model."""
    
    def __init__(self, data_path: str = None, output_dir: str = None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the fertilizer dataset CSV
            output_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / 'models' / 'fertilizer'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.soil_encoder = LabelEncoder()
        self.crop_encoder = LabelEncoder()
        self.fertilizer_encoder = LabelEncoder()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the fertilizer dataset."""
        if self.data_path and os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset from {self.data_path}")
        else:
            # Generate synthetic data for training
            logger.info("Generating synthetic fertilizer dataset...")
            df = self._generate_synthetic_data()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        logger.info(f"Dataset shape: {df.shape}")
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic fertilizer data for training."""
        np.random.seed(42)
        n_samples = 2000
        
        # Define categories
        soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
        crop_types = ['Wheat', 'Cotton', 'Maize', 'Paddy', 'Barley', 
                      'Ground Nuts', 'Sugarcane', 'Potato', 'Tomato', 'Onion']
        fertilizers = ['Urea', 'DAP', 'MOP', 'NPK-10-26-26', 'NPK-20-20-20', 
                       'SSP', 'Ammonium Sulphate', 'Super Phosphate', '14-35-14', '10-10-10']
        
        data = []
        
        for _ in range(n_samples):
            soil = np.random.choice(soil_types)
            crop = np.random.choice(crop_types)
            
            # Generate realistic environmental conditions
            temperature = np.random.uniform(15, 40)
            humidity = np.random.uniform(30, 90)
            moisture = np.random.uniform(20, 80)
            
            # Generate nutrient levels based on soil type
            if soil == 'Black':
                n = np.random.randint(40, 120)
                p = np.random.randint(30, 80)
                k = np.random.randint(40, 100)
            elif soil == 'Red':
                n = np.random.randint(20, 80)
                p = np.random.randint(15, 50)
                k = np.random.randint(30, 80)
            elif soil == 'Sandy':
                n = np.random.randint(10, 60)
                p = np.random.randint(10, 40)
                k = np.random.randint(20, 60)
            else:  # Loamy, Clayey
                n = np.random.randint(30, 100)
                p = np.random.randint(20, 70)
                k = np.random.randint(30, 90)
            
            # Determine fertilizer based on rules
            fertilizer = self._determine_fertilizer(n, p, k, crop, soil)
            
            data.append({
                'Temparature': temperature,
                'Humidity ': humidity,
                'Moisture': moisture,
                'Soil Type': soil,
                'Crop Type': crop,
                'Nitrogen': n,
                'Potassium': k,
                'Phosphorous': p,
                'Fertilizer': fertilizer
            })
        
        return pd.DataFrame(data)
    
    def _determine_fertilizer(self, n: int, p: int, k: int, crop: str, soil: str) -> str:
        """Determine appropriate fertilizer based on nutrient levels and crop."""
        # Crop nutrient requirements
        crop_needs = {
            'Wheat': {'N': 120, 'P': 60, 'K': 40},
            'Cotton': {'N': 90, 'P': 45, 'K': 45},
            'Maize': {'N': 150, 'P': 75, 'K': 60},
            'Paddy': {'N': 100, 'P': 50, 'K': 50},
            'Barley': {'N': 80, 'P': 40, 'K': 30},
            'Ground Nuts': {'N': 20, 'P': 40, 'K': 40},
            'Sugarcane': {'N': 250, 'P': 125, 'K': 100},
            'Potato': {'N': 150, 'P': 100, 'K': 150},
            'Tomato': {'N': 100, 'P': 80, 'K': 80},
            'Onion': {'N': 100, 'P': 60, 'K': 60}
        }
        
        needs = crop_needs.get(crop, {'N': 100, 'P': 50, 'K': 50})
        
        n_gap = needs['N'] - n
        p_gap = needs['P'] - p
        k_gap = needs['K'] - k
        
        # Select fertilizer based on gaps
        if n_gap <= 0 and p_gap <= 0 and k_gap <= 0:
            return 'NPK-20-20-20'
        elif n_gap > p_gap and n_gap > k_gap:
            return 'Urea' if p_gap < 20 else 'DAP'
        elif p_gap > k_gap:
            return 'DAP' if n_gap > 20 else 'SSP'
        else:
            return 'MOP'
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess the data for training."""
        # Encode categorical variables
        df['Soil Type Encoded'] = self.soil_encoder.fit_transform(df['Soil Type'])
        df['Crop Type Encoded'] = self.crop_encoder.fit_transform(df['Crop Type'])
        df['Fertilizer Encoded'] = self.fertilizer_encoder.fit_transform(df['Fertilizer'])
        
        # Feature columns (matching the expected input format)
        feature_cols = ['Temparature', 'Humidity ', 'Moisture', 
                        'Soil Type Encoded', 'Crop Type Encoded',
                        'Nitrogen', 'Potassium', 'Phosphorous']
        
        X = df[feature_cols]
        y = df['Fertilizer Encoded']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(self.X_train)}")
        logger.info(f"Testing samples: {len(self.X_test)}")
        logger.info(f"Number of classes: {len(self.fertilizer_encoder.classes_)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self, model_type: str = 'random_forest') -> object:
        """
        Train the fertilizer recommendation model.
        
        Args:
            model_type: Type of model to train ('random_forest' or 'gradient_boosting')
            
        Returns:
            Trained model
        """
        config = FERTILIZER_CONFIG.get(model_type, FERTILIZER_CONFIG['random_forest'])
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 15),
                min_samples_split=config.get('min_samples_split', 5),
                min_samples_leaf=config.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                learning_rate=config.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Training {model_type} model...")
        self.model.fit(self.X_train, self.y_train)
        
        return self.model
    
    def evaluate(self) -> dict:
        """Evaluate the trained model."""
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info("=" * 50)
        logger.info("Model Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        logger.info("=" * 50)
        
        # Classification report
        class_names = self.fertilizer_encoder.classes_
        report = classification_report(
            self.y_test, y_pred, 
            target_names=class_names,
            zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")
        
        return metrics
    
    def save_models(self):
        """Save the trained model and encoders."""
        # Save model
        model_path = self.output_dir / 'fertilizer_model.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save encoders
        joblib.dump(self.soil_encoder, self.output_dir / 'soil_encoder.pkl')
        joblib.dump(self.crop_encoder, self.output_dir / 'crop_encoder.pkl')
        joblib.dump(self.fertilizer_encoder, self.output_dir / 'fertilizer_encoder.pkl')
        
        logger.info(f"Encoders saved to {self.output_dir}")
        
        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'model_type': type(self.model).__name__,
            'soil_types': list(self.soil_encoder.classes_),
            'crop_types': list(self.crop_encoder.classes_),
            'fertilizer_types': list(self.fertilizer_encoder.classes_),
            'feature_columns': list(self.X_train.columns)
        }
        
        import json
        with open(self.output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {self.output_dir / 'model_metadata.json'}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Fertilizer Recommendation Model')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to fertilizer dataset CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for trained models')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting'],
                        help='Model type to train')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FertilizerModelTrainer(
        data_path=args.data,
        output_dir=args.output
    )
    
    # Load and preprocess data
    df = trainer.load_data()
    trainer.preprocess_data(df)
    
    # Train model
    trainer.train(model_type=args.model)
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Save models
    trainer.save_models()
    
    logger.info("Training completed successfully!")
    
    return metrics


if __name__ == '__main__':
    main()
