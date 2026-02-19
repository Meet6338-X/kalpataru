"""
CNN Model for Plant Disease Detection.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from config.settings import IMAGE_SIZE, DISEASE_CLASSES

def build_disease_model(num_classes: int = None) -> Sequential:
    """
    Build CNN model for disease detection.
    
    Args:
        num_classes: Number of disease classes
        
    Returns:
        Compiled CNN model
    """
    if num_classes is None:
        num_classes = len(DISEASE_CLASSES)
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Dense layers
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_pretrained_model(model_path: str) -> Sequential:
    """
    Load pretrained disease detection model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    model = build_disease_model()
    model.load_weights(model_path)
    return model
