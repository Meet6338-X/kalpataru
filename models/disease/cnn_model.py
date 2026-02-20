"""
CNN Model for Plant Disease Detection.
Supports both TensorFlow/Keras and PyTorch models for 38 disease classes.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from config.settings import IMAGE_SIZE
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Extended disease classes (38 classes from PlantVillage dataset)
DISEASE_CLASSES_38 = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease descriptions and treatment recommendations
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'Fungal disease causing dark, scaly lesions on leaves and fruit.',
        'treatment': 'Apply fungicides (captan, mancozeb) during wet periods. Remove fallen leaves.',
        'prevention': 'Plant resistant varieties. Ensure good air circulation.'
    },
    'Apple___Black_rot': {
        'description': 'Fungal disease causing leaf spots and fruit rot.',
        'treatment': 'Prune infected branches. Apply fungicides during bloom.',
        'prevention': 'Remove mummified fruit. Maintain tree vigor.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Fungal disease causing orange spots on leaves and fruit.',
        'treatment': 'Apply fungicides (myclobutanil) when symptoms appear.',
        'prevention': 'Remove nearby cedar trees if possible.'
    },
    'Apple___healthy': {
        'description': 'The apple plant is healthy with no visible disease symptoms.',
        'treatment': 'Continue regular plant care and monitoring.',
        'prevention': 'Maintain good cultural practices.'
    },
    'Blueberry___healthy': {
        'description': 'The blueberry plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Maintain acidic soil pH (4.5-5.5).'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Fungal disease causing white powdery growth on leaves.',
        'treatment': 'Apply sulfur or potassium bicarbonate fungicides.',
        'prevention': 'Improve air circulation. Avoid overhead irrigation.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'The cherry plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Fungal disease causing rectangular gray lesions on leaves.',
        'treatment': 'Apply fungicides if infection is severe.',
        'prevention': 'Use resistant hybrids. Rotate crops.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Fungal disease causing reddish-brown pustules on leaves.',
        'treatment': 'Apply fungicides (propiconazole) if severe.',
        'prevention': 'Plant resistant varieties. Avoid late planting.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Fungal disease causing long, cigar-shaped lesions.',
        'treatment': 'Apply fungicides during early infection.',
        'prevention': 'Use resistant hybrids. Crop rotation.'
    },
    'Corn_(maize)___healthy': {
        'description': 'The corn plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Grape___Black_rot': {
        'description': 'Fungal disease causing black spots on leaves and fruit.',
        'treatment': 'Apply fungicides (mancozeb, myclobutanil) every 7-14 days.',
        'prevention': 'Remove mummified fruit. Ensure good air circulation.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Fungal disease causing "tiger stripe" pattern on leaves.',
        'treatment': 'No effective cure. Remove severely infected vines.',
        'prevention': 'Avoid trunk injuries. Proper pruning practices.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Fungal disease causing angular leaf spots.',
        'treatment': 'Apply copper-based fungicides.',
        'prevention': 'Remove infected leaves. Improve air circulation.'
    },
    'Grape___healthy': {
        'description': 'The grape plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Bacterial disease causing yellow shoots and misshapen fruit.',
        'treatment': 'No cure. Remove infected trees to prevent spread.',
        'prevention': 'Control Asian citrus psyllid vector. Use disease-free nursery stock.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial disease causing angular leaf spots and fruit lesions.',
        'treatment': 'Apply copper-based bactericides.',
        'prevention': 'Use resistant varieties. Avoid overhead irrigation.'
    },
    'Peach___healthy': {
        'description': 'The peach plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial disease causing water-soaked lesions on leaves and fruit.',
        'treatment': 'Apply copper-based bactericides.',
        'prevention': 'Use disease-free seeds. Avoid working with wet plants.'
    },
    'Pepper,_bell___healthy': {
        'description': 'The bell pepper plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Potato___Early_blight': {
        'description': 'Fungal disease causing target-like lesions on leaves.',
        'treatment': 'Apply fungicides (chlorothalonil, mancozeb).',
        'prevention': 'Crop rotation. Remove infected plant debris.'
    },
    'Potato___Late_blight': {
        'description': 'Fungal disease causing water-soaked lesions and tuber rot.',
        'treatment': 'Apply fungicides preventatively. Destroy infected plants.',
        'prevention': 'Use resistant varieties. Ensure good drainage.'
    },
    'Potato___healthy': {
        'description': 'The potato plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Raspberry___healthy': {
        'description': 'The raspberry plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Prune properly. Ensure good air circulation.'
    },
    'Soybean___healthy': {
        'description': 'The soybean plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Squash___Powdery_mildew': {
        'description': 'Fungal disease causing white powdery growth on leaves.',
        'treatment': 'Apply sulfur or neem oil based fungicides.',
        'prevention': 'Improve air circulation. Avoid overhead watering.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Fungal disease causing purple-red leaf spots.',
        'treatment': 'Apply fungicides (myclobutanil).',
        'prevention': 'Remove infected leaves. Ensure good drainage.'
    },
    'Strawberry___healthy': {
        'description': 'The strawberry plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial disease causing small, water-soaked spots on leaves and fruit.',
        'treatment': 'Apply copper-based bactericides. Remove infected leaves.',
        'prevention': 'Use disease-free seeds. Avoid overhead irrigation.'
    },
    'Tomato___Early_blight': {
        'description': 'Fungal disease causing target-like lesions on lower leaves.',
        'treatment': 'Apply fungicides (chlorothalonil). Remove infected leaves.',
        'prevention': 'Mulch around plants. Ensure good air circulation.'
    },
    'Tomato___Late_blight': {
        'description': 'Fungal disease causing water-soaked lesions and fruit rot.',
        'treatment': 'Apply fungicides preventatively. Remove infected plants.',
        'prevention': 'Use resistant varieties. Avoid wetting foliage.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Fungal disease causing yellow spots on upper leaf surface.',
        'treatment': 'Apply fungicides. Increase ventilation.',
        'prevention': 'Reduce humidity. Space plants properly.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Fungal disease causing circular spots with dark borders.',
        'treatment': 'Apply fungicides (chlorothalonil). Remove infected leaves.',
        'prevention': 'Mulch around plants. Water at base.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Pest causing stippling and yellowing of leaves.',
        'treatment': 'Apply insecticidal soap or neem oil. Introduce predatory mites.',
        'prevention': 'Increase humidity. Remove weeds.'
    },
    'Tomato___Target_Spot': {
        'description': 'Fungal disease causing target-like lesions on leaves.',
        'treatment': 'Apply fungicides (chlorothalonil, mancozeb).',
        'prevention': 'Improve air circulation. Avoid overhead irrigation.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Viral disease causing yellowing and curling of leaves.',
        'treatment': 'No cure. Remove infected plants.',
        'prevention': 'Control whiteflies. Use resistant varieties.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Viral disease causing mosaic pattern on leaves.',
        'treatment': 'No cure. Remove infected plants.',
        'prevention': 'Use virus-free seeds. Control aphids. Disinfect tools.'
    },
    'Tomato___healthy': {
        'description': 'The tomato plant is healthy.',
        'treatment': 'Continue regular care.',
        'prevention': 'Monitor for pests and diseases.'
    }
}


def build_disease_model(num_classes: int = 38, model_type: str = 'custom'):
    """
    Build CNN model for disease detection.
    
    Args:
        num_classes: Number of disease classes (default 38 for full PlantVillage)
        model_type: Type of model ('custom', 'mobilenet', 'resnet')
        
    Returns:
        Compiled model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Conv2D, MaxPooling2D, Dense, Flatten, 
            Dropout, BatchNormalization, GlobalAveragePooling2D
        )
        from tensorflow.keras.optimizers import Adam
        
        if model_type == 'custom':
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
                
                # Dense layers
                GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(num_classes, activation='softmax')
            ])
            
        elif model_type == 'mobilenet':
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import Model, Input
            
            base = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*IMAGE_SIZE, 3)
            )
            base.trainable = False
            
            inputs = Input(shape=(*IMAGE_SIZE, 3))
            x = base(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs, outputs)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except ImportError:
        logger.warning("TensorFlow not available. Model building disabled.")
        return None


def load_pretrained_model(model_path: str, num_classes: int = 38):
    """
    Load pretrained disease detection model.
    
    Args:
        model_path: Path to saved model
        num_classes: Number of disease classes
        
    Returns:
        Loaded model
    """
    path = Path(model_path)
    
    if not path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        # Try loading as TensorFlow/Keras model
        import tensorflow as tf
        
        if path.suffix in ['.h5', '.keras']:
            model = tf.keras.models.load_model(str(path))
            logger.info(f"Loaded Keras model from {model_path}")
            return model
        else:
            # Build model and load weights
            model = build_disease_model(num_classes)
            model.load_weights(str(path))
            logger.info(f"Loaded model weights from {model_path}")
            return model
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def load_pytorch_model(model_path: str, num_classes: int = 38):
    """
    Load PyTorch disease detection model.
    
    Args:
        model_path: Path to saved PyTorch model (.pth or .pt)
        num_classes: Number of disease classes
        
    Returns:
        Loaded PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        
        class PlantDiseaseNet(nn.Module):
            """PyTorch CNN for plant disease classification."""
            
            def __init__(self, num_classes=38):
                super(PlantDiseaseNet, self).__init__()
                
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Block 2
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Block 3
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Block 4
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = PlantDiseaseNet(num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        logger.info(f"Loaded PyTorch model from {model_path}")
        return model
        
    except ImportError:
        logger.warning("PyTorch not available. Cannot load PyTorch model.")
        return None
    except Exception as e:
        logger.error(f"Error loading PyTorch model: {str(e)}")
        return None


def get_disease_info(disease_name: str) -> Dict[str, str]:
    """
    Get information about a specific disease.
    
    Args:
        disease_name: Name of the disease
        
    Returns:
        Dictionary with disease information
    """
    return DISEASE_INFO.get(disease_name, {
        'description': 'Information not available for this disease.',
        'treatment': 'Consult local agricultural expert.',
        'prevention': 'Follow general plant care practices.'
    })


def get_all_disease_classes() -> List[str]:
    """Get list of all supported disease classes."""
    return DISEASE_CLASSES_38.copy()
