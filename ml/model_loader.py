"""
Model Loader for Plant Disease Detection
Loads MobileNetV2 model with custom classifier for 38 plant disease classes.
"""
import os
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


class PlantDiseaseModel:
    """Plant Disease Detection Model using MobileNetV2"""

    def __init__(self, model_path: str = None):
        """
        Initialize the model

        Args:
            model_path: Path to the model weights file (.pth state dict)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the MobileNetV2 model with custom classifier"""
        loaded = False
        
        # Create the base model architecture
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.classifier[1].in_features, 38)
        )
        
        # Try loading state dict from .pth file
        if self.model_path and os.path.exists(self.model_path):
            # Check if it's a file (not directory)
            if os.path.isfile(self.model_path):
                try:
                    state_dict = torch.load(
                        self.model_path, 
                        map_location=self.device, 
                        weights_only=False
                    )
                    self.model.load_state_dict(state_dict)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"[OK] Loaded trained model from {self.model_path}")
                    loaded = True
                except Exception as e:
                    print(f"[WARN] Could not load model weights: {e}")
            else:
                # It's a directory - try TorchScript format
                try:
                    self.model = torch.jit.load(self.model_path, map_location=self.device)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"[OK] Loaded TorchScript model from {self.model_path}")
                    loaded = True
                except Exception as e:
                    print(f"[WARN] Could not load TorchScript model: {e}")

        # Fallback to pretrained MobileNetV2
        if not loaded:
            print("[INFO] Using pretrained MobileNetV2 - predictions may not be accurate")
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

            # Freeze feature extractor layers
            for param in self.model.features.parameters():
                param.requires_grad = False

            # Replace the classifier with custom 38-class classifier
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.classifier[1].in_features, 38)
            )

            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Make prediction on an image tensor

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)

        Returns:
            Prediction logits
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
        return output

    def get_probabilities(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            Tensor of class probabilities
        """
        output = self.predict(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        return probabilities


# Singleton instance
_model_instance = None


def get_model(model_path: str = None) -> PlantDiseaseModel:
    """
    Get singleton model instance

    Args:
        model_path: Optional path to model weights

    Returns:
        PlantDiseaseModel instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = PlantDiseaseModel(model_path)
    return _model_instance


def reset_model():
    """Reset the singleton model instance (useful for testing)."""
    global _model_instance
    _model_instance = None
