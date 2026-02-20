"""
Training script for Soil Classification Model.
Trains a CNN model to classify soil types from images.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from config.training_config import SOIL_CONFIG

logger = setup_logger(__name__)

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Some features will be disabled.")


class SoilClassifierTrainer:
    """Trainer class for soil classification CNN model."""
    
    # Soil types for classification
    SOIL_TYPES = ['Alluvial', 'Black', 'Clay', 'Laterite', 'Red', 'Sandy']
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Path to soil images dataset directory
            output_dir: Directory to save trained models
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for training soil classifier")
        
        self.data_dir = Path(data_dir) if data_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / 'models' / 'soil'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_size = SOIL_CONFIG.get('image_size', (224, 224))
        self.batch_size = SOIL_CONFIG.get('batch_size', 32)
        self.epochs = SOIL_CONFIG.get('epochs', 50)
        
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.model = None
    
    def prepare_data_generators(self) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        """
        Prepare data generators for training, validation, and testing.
        
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        if self.data_dir is None or not self.data_dir.exists():
            logger.warning("Data directory not found. Creating synthetic dataset...")
            self._create_synthetic_dataset()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation and test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            str(self.data_dir / 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        self.val_generator = train_datagen.flow_from_directory(
            str(self.data_dir / 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Test generator
        test_dir = self.data_dir / 'test'
        if test_dir.exists():
            self.test_generator = test_datagen.flow_from_directory(
                str(test_dir),
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        logger.info(f"Training samples: {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.val_generator.samples}")
        if self.test_generator:
            logger.info(f"Test samples: {self.test_generator.samples}")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def _create_synthetic_dataset(self):
        """Create a synthetic dataset for demonstration."""
        logger.info("Creating synthetic soil image dataset...")
        
        self.data_dir = Path(__file__).parent.parent / 'data' / 'synthetic_soil'
        
        for split in ['train', 'test']:
            for soil_type in self.SOIL_TYPES:
                soil_dir = self.data_dir / split / soil_type
                soil_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic images
        np.random.seed(42)
        
        for soil_type in self.SOIL_TYPES:
            # Define color ranges for each soil type
            color_ranges = {
                'Alluvial': ((139, 115, 85), (210, 180, 140)),
                'Black': ((28, 28, 28), (74, 74, 74)),
                'Clay': ((139, 69, 19), (160, 82, 45)),
                'Laterite': ((178, 34, 34), (205, 92, 92)),
                'Red': ((205, 92, 92), (240, 128, 128)),
                'Sandy': ((244, 164, 96), (250, 235, 215))
            }
            
            low, high = color_ranges[soil_type]
            
            for split in ['train', 'test']:
                n_samples = 100 if split == 'train' else 20
                
                for i in range(n_samples):
                    # Create base color
                    r = np.random.randint(low[0], high[0] + 1)
                    g = np.random.randint(low[1], high[1] + 1)
                    b = np.random.randint(low[2], high[2] + 1)
                    
                    # Create image with noise
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                    noise = np.random.randint(-20, 21, (224, 224, 3))
                    
                    img[:, :, 0] = np.clip(r + noise[:, :, 0], 0, 255)
                    img[:, :, 1] = np.clip(g + noise[:, :, 1], 0, 255)
                    img[:, :, 2] = np.clip(b + noise[:, :, 2], 0, 255)
                    
                    # Save image
                    from PIL import Image
                    img_path = self.data_dir / split / soil_type / f'{soil_type}_{i}.jpg'
                    Image.fromarray(img).save(img_path)
        
        logger.info(f"Synthetic dataset created at {self.data_dir}")
    
    def build_model(self, model_type: str = 'custom') -> tf.keras.Model:
        """
        Build the CNN model for soil classification.
        
        Args:
            model_type: Type of model architecture ('custom', 'mobilenet', 'resnet')
            
        Returns:
            Compiled Keras model
        """
        num_classes = len(self.SOIL_TYPES)
        
        if model_type == 'custom':
            self.model = self._build_custom_cnn(num_classes)
        elif model_type == 'mobilenet':
            self.model = self._build_transfer_model('mobilenet', num_classes)
        elif model_type == 'resnet':
            self.model = self._build_transfer_model('resnet', num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model
    
    def _build_custom_cnn(self, num_classes: int) -> tf.keras.Model:
        """Build a custom CNN architecture."""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_transfer_model(self, base_model: str, num_classes: int) -> tf.keras.Model:
        """Build a transfer learning model."""
        if base_model == 'mobilenet':
            from tensorflow.keras.applications import MobileNetV2
            base = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        elif base_model == 'resnet':
            from tensorflow.keras.applications import ResNet50
            base = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Freeze base model layers
        base.trainable = False
        
        # Build custom top
        inputs = layers.Input(shape=(*self.image_size, 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, model_type: str = 'custom') -> dict:
        """
        Train the soil classification model.
        
        Args:
            model_type: Type of model architecture
            
        Returns:
            Training history
        """
        # Prepare data
        self.prepare_data_generators()
        
        # Build model
        self.build_model(model_type)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                str(self.output_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        logger.info(f"Starting training for {self.epochs} epochs...")
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks
        )
        
        return history.history
    
    def evaluate(self) -> dict:
        """Evaluate the trained model."""
        if self.test_generator is None:
            logger.warning("No test generator available. Skipping evaluation.")
            return {}
        
        results = self.model.evaluate(self.test_generator)
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1]
        }
        
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]:.4f}")
        
        return metrics
    
    def save_model(self):
        """Save the trained model and metadata."""
        # Save model
        model_path = self.output_dir / 'soil_classifier_model.h5'
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save class indices
        class_indices = {v: k for k, v in self.train_generator.class_indices.items()}
        
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'model_type': type(self.model).__name__,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'epochs_trained': self.epochs,
            'soil_types': self.SOIL_TYPES,
            'class_indices': class_indices
        }
        
        with open(self.output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {self.output_dir / 'model_metadata.json'}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Soil Classification Model')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to soil images dataset directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for trained models')
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'mobilenet', 'resnet'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SoilClassifierTrainer(
        data_dir=args.data,
        output_dir=args.output
    )
    
    if args.epochs:
        trainer.epochs = args.epochs
    
    # Train model
    history = trainer.train(model_type=args.model)
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Save model
    trainer.save_model()
    
    logger.info("Training completed successfully!")
    
    return {**history, **metrics}


if __name__ == '__main__':
    main()
