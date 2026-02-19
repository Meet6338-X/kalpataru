"""
Training Script for Plant Disease Detection CNN Model.

This script trains a CNN model for plant disease detection using the preprocessed images.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / 'disease_images'
MODEL_DIR = BASE_DIR / 'models' / 'disease'
ANALYSIS_DIR = BASE_DIR / 'data' / 'analysis' / 'disease_detection'

# Training settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


def setup_gpu():
    """Setup GPU configuration."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.info("No GPU found, using CPU")


def load_metadata():
    """Load dataset metadata."""
    metadata_path = PROCESSED_DIR / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded metadata: {metadata['num_classes']} classes")
    logger.info(f"Total images: {metadata['totals']['all']}")
    
    return metadata


def create_data_generators():
    """Create data generators for training, validation, and testing."""
    logger.info("Creating data generators...")
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test generators (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        PROCESSED_DIR / 'train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        PROCESSED_DIR / 'val',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        PROCESSED_DIR / 'test',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info(f"Training samples: {train_generator.samples}")
    logger.info(f"Validation samples: {val_generator.samples}")
    logger.info(f"Test samples: {test_generator.samples}")
    logger.info(f"Number of classes: {train_generator.num_classes}")
    
    return train_generator, val_generator, test_generator


def build_custom_cnn(num_classes):
    """Build a custom CNN model."""
    logger.info("Building custom CNN model...")
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Fifth convolutional block
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Dense layers
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built with {model.count_params()} parameters")
    
    return model


def build_transfer_learning_model(num_classes, base_model_name='mobilenet'):
    """Build a transfer learning model using pre-trained weights."""
    logger.info(f"Building transfer learning model with {base_model_name}...")
    
    # Select base model
    if base_model_name == 'mobilenet':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3)
        )
    elif base_model_name == 'efficientnet':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3)
        )
    elif base_model_name == 'resnet':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3)
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built with {model.count_params()} parameters")
    
    return model, base_model


def unfreeze_layers(model, base_model, num_layers_to_unfreeze=20):
    """Unfreeze some layers for fine-tuning."""
    logger.info(f"Unfreezing last {num_layers_to_unfreeze} layers...")
    
    base_model.trainable = True
    
    # Freeze all but the last N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_name):
    """Get training callbacks."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=str(MODEL_DIR / f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, train_generator, val_generator, callbacks, epochs=EPOCHS):
    """Train the model."""
    logger.info("Starting training...")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_generator):
    """Evaluate the model on test data."""
    logger.info("Evaluating model on test data...")
    
    # Reset generator
    test_generator.reset()
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Classification metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    class_indices = test_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(class_indices))]
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, y_pred, y_true


def save_model_and_results(model, history, metrics, class_indices, model_name='disease_detection'):
    """Save model and training results."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model weights
    weights_path = MODEL_DIR / f'{model_name}_weights.h5'
    model.save_weights(str(weights_path))
    logger.info(f"Saved model weights to {weights_path}")
    
    # Save full model
    model_path = MODEL_DIR / f'{model_name}_model.h5'
    model.save(str(model_path))
    logger.info(f"Saved full model to {model_path}")
    
    # Save class indices
    indices_path = MODEL_DIR / f'{model_name}_class_indices.pkl'
    with open(indices_path, 'wb') as f:
        pickle.dump(class_indices, f)
    logger.info(f"Saved class indices to {indices_path}")
    
    # Save training history
    history_path = ANALYSIS_DIR / f'{model_name}_history_{timestamp}.json'
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    # Save metrics
    metrics_path = ANALYSIS_DIR / f'{model_name}_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")


def print_classification_report(metrics, class_indices):
    """Print detailed classification report."""
    logger.info("\n" + "=" * 60)
    logger.info("Classification Report")
    logger.info("=" * 60)
    
    report = metrics['classification_report']
    
    # Print per-class metrics
    logger.info("\nPer-class metrics:")
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    for i in range(len(class_indices)):
        class_name = idx_to_class[i]
        if str(i) in report:
            class_metrics = report[str(i)]
            logger.info(f"  {class_name}: Precision={class_metrics['precision']:.3f}, "
                       f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Plant Disease Detection Model Training")
    logger.info("=" * 60)
    
    # Setup GPU
    setup_gpu()
    
    # Step 1: Load metadata
    logger.info("\n[Step 1/5] Loading metadata...")
    metadata = load_metadata()
    num_classes = metadata['num_classes']
    
    # Step 2: Create data generators
    logger.info("\n[Step 2/5] Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators()
    
    # Step 3: Build model
    logger.info("\n[Step 3/5] Building model...")
    
    # Option 1: Custom CNN
    # model = build_custom_cnn(num_classes)
    # model_name = 'disease_custom_cnn'
    
    # Option 2: Transfer Learning (recommended)
    model, base_model = build_transfer_learning_model(num_classes, base_model_name='mobilenet')
    model_name = 'disease_mobilenet'
    
    # Print model summary
    model.summary()
    
    # Step 4: Train model
    logger.info("\n[Step 4/5] Training model...")
    callbacks = get_callbacks(model_name)
    
    # Phase 1: Train with frozen base
    logger.info("Phase 1: Training with frozen base model...")
    history = train_model(model, train_generator, val_generator, callbacks, epochs=30)
    
    # Phase 2: Fine-tune (optional)
    logger.info("Phase 2: Fine-tuning...")
    model = unfreeze_layers(model, base_model, num_layers_to_unfreeze=20)
    history_fine = train_model(model, train_generator, val_generator, callbacks, epochs=20)
    
    # Combine histories
    for key in history.history:
        history.history[key].extend(history_fine.history[key])
    
    # Step 5: Evaluate model
    logger.info("\n[Step 5/5] Evaluating model...")
    metrics, y_pred, y_true = evaluate_model(model, test_generator)
    
    # Save model and results
    save_model_and_results(model, history, metrics, test_generator.class_indices, model_name)
    
    # Print classification report
    print_classification_report(metrics, test_generator.class_indices)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    logger.info(f"\nTest accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Model saved to: {MODEL_DIR}")


if __name__ == '__main__':
    main()