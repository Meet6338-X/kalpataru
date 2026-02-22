"""
Image Preprocessing for Plant Disease Detection
Handles image loading, resizing, and tensor conversion
"""
import io
from PIL import Image
import torch
from torchvision import transforms


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms() -> transforms.Compose:
    """
    Get the image transformation pipeline

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load an image from bytes

    Args:
        image_bytes: Image data as bytes

    Returns:
        PIL Image
    """
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess image for model inference

    Args:
        image_bytes: Image data as bytes

    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = load_image_from_bytes(image_bytes)

    # Get transforms
    transform = get_transforms()

    # Apply transforms
    tensor = transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


def preprocess_multiple_images(image_bytes_list: list) -> torch.Tensor:
    """
    Preprocess multiple images for batch inference

    Args:
        image_bytes_list: List of image bytes

    Returns:
        Batched image tensor
    """
    transform = get_transforms()
    tensors = []

    for image_bytes in image_bytes_list:
        image = load_image_from_bytes(image_bytes)
        tensor = transform(image)
        tensors.append(tensor)

    return torch.stack(tensors)
