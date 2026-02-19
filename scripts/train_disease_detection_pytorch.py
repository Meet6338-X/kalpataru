"""
Training Script for Plant Disease Detection CNN Model using PyTorch.

This script trains a CNN model for plant disease detection using the preprocessed images.
Features:
- GPU acceleration support
- Checkpoint saving after each epoch
- Saves BEST model based on highest validation accuracy
- Automatic resume capability
- Early stopping to prevent overfitting
"""

import os
import json
import logging
import pickle
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / 'disease_images'
MODEL_DIR = BASE_DIR / 'models' / 'disease'
ANALYSIS_DIR = BASE_DIR / 'data' / 'analysis' / 'disease_detection'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'disease'

# Training settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0001
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs

# Device configuration with detailed logging
def get_device():
    """Get the best available device with detailed logging."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_capability = torch.cuda.get_device_capability(0)
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"GPU memory: {gpu_memory:.2f} GB")
        logger.info(f"GPU Compute Capability: sm_{gpu_capability[0]}{gpu_capability[1]}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        
        # Check if GPU compute capability is supported
        # Blackwell (sm_120) and newer architectures may not be supported yet
        supported_capabilities = [(5, 0), (6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0)]
        if gpu_capability not in supported_capabilities:
            logger.warning(f"GPU compute capability sm_{gpu_capability[0]}{gpu_capability[1]} may not be fully supported.")
            logger.warning("Falling back to CPU for compatibility.")
            device = torch.device('cpu')
            logger.info(f"Using device: {device}")
        else:
            logger.info(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        logger.warning("No GPU detected! Training on CPU will be much slower.")
        logger.info(f"Using device: {device}")
    return device

device = get_device()


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


def create_data_loaders():
    """Create data loaders for training, validation, and testing."""
    logger.info("Creating data loaders...")
    
    # Data transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data transforms for validation and testing (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        PROCESSED_DIR / 'train', transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        PROCESSED_DIR / 'val', transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        PROCESSED_DIR / 'test', transform=val_transform
    )
    
    # Create data loaders with pin_memory for GPU acceleration
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=pin_memory
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes):
    """Build a MobileNetV2 model with transfer learning."""
    logger.info("Building MobileNetV2 model with transfer learning...")
    
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Freeze the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace the classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, phase, history, best_val_acc, path, is_best=False):
    """Save training checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'phase': phase,  # 'train' or 'finetune'
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_val_acc': best_val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")
    
    # Save best model separately
    if is_best:
        best_model_path = MODEL_DIR / 'disease_mobilenet_best.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': best_val_acc,
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat()
        }, best_model_path)
        logger.info(f"*** BEST MODEL SAVED! Val Acc: {best_val_acc:.2f}% -> {best_model_path} ***")


def save_intermediate_model(model, epoch, phase, val_acc):
    """Save intermediate model during training."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    intermediate_path = MODEL_DIR / f'disease_mobilenet_epoch{epoch+1}_{phase}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch + 1,
        'phase': phase,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat()
    }, intermediate_path)
    logger.info(f"Intermediate model saved: {intermediate_path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (
        checkpoint['epoch'],
        checkpoint['phase'],
        checkpoint['history'],
        checkpoint['best_val_acc']
    )


def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, epochs, learning_rate, fine_tune=False, 
                start_epoch=0, existing_history=None, existing_best_acc=0.0, checkpoint_path=None):
    """Train the model with checkpoint support and early stopping."""
    criterion = nn.CrossEntropyLoss()
    
    if fine_tune:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_acc = existing_best_acc
    best_model_state = None
    history = existing_history or {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Early stopping variables
    epochs_without_improvement = 0
    
    phase = "Fine-tuning" if fine_tune else "Training"
    logger.info(f"\n{phase} for {epochs} epochs...")
    if start_epoch > 0:
        logger.info(f"Resuming from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            logger.info(f"  -> New best model! Val Acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            logger.info(f"  -> No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save checkpoint after each epoch
        if checkpoint_path:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, 
                'finetune' if fine_tune else 'train',
                history, best_val_acc, checkpoint_path, is_best=is_best
            )
        
        # Save intermediate model every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_intermediate_model(model, epoch, 'finetune' if fine_tune else 'train', val_acc)
        
        # Early stopping check
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logger.info(f"\nEarly stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model with Val Acc: {best_val_acc:.2f}%")
    
    return model, history, best_val_acc


def unfreeze_model(model, num_layers=20):
    """Unfreeze the last N layers for fine-tuning."""
    logger.info(f"Unfreezing last {num_layers} layers...")
    
    # Unfreeze all parameters first
    for param in model.parameters():
        param.requires_grad = True
    
    # Freeze all but the last N layers
    features = list(model.features.children())
    for i, layer in enumerate(features[:-num_layers]):
        for param in layer.parameters():
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters after unfreezing: {trainable:,}")
    
    return model


def evaluate_model(model, test_loader, class_names):
    """Evaluate the model on test data."""
    logger.info("\nEvaluating model on test data...")
    
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100.0 * correct / total
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'test_accuracy': float(test_acc / 100),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, all_preds, all_labels


def save_final_model(model, history, metrics, class_names, class_to_idx, best_val_acc):
    """Save final model and all training results."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model weights
    weights_path = MODEL_DIR / 'disease_mobilenet_weights.pt'
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Saved model weights to {weights_path}")
    
    # Save full model with all metadata
    model_path = MODEL_DIR / 'disease_mobilenet_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'metrics': metrics,
        'best_val_acc': best_val_acc,
        'timestamp': timestamp
    }, model_path)
    logger.info(f"Saved full model to {model_path}")
    
    # Save class indices
    indices_path = MODEL_DIR / 'disease_detection_class_indices.pkl'
    with open(indices_path, 'wb') as f:
        pickle.dump(class_to_idx, f)
    logger.info(f"Saved class indices to {indices_path}")
    
    # Save training history
    history_path = ANALYSIS_DIR / f'disease_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    # Save metrics
    metrics_path = ANALYSIS_DIR / f'disease_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save a summary file
    summary_path = MODEL_DIR / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PLANT DISEASE DETECTION MODEL - TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test Accuracy: {metrics['test_accuracy']*100:.2f}%\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"\nNumber of classes: {len(class_names)}\n")
        f.write(f"Total epochs trained: {len(history['train_loss'])}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("FILES SAVED:\n")
        f.write("=" * 60 + "\n")
        f.write(f"  - Model: {model_path}\n")
        f.write(f"  - Weights: {weights_path}\n")
        f.write(f"  - Best Model: {MODEL_DIR / 'disease_mobilenet_best.pt'}\n")
        f.write(f"  - Class Indices: {indices_path}\n")
        f.write(f"  - History: {history_path}\n")
        f.write(f"  - Metrics: {metrics_path}\n")
    logger.info(f"Saved training summary to {summary_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--fine-tune-epochs', type=int, default=FINE_TUNE_EPOCHS,
                        help=f'Number of fine-tuning epochs (default: {FINE_TUNE_EPOCHS})')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Plant Disease Detection Model Training (PyTorch)")
    logger.info("=" * 60)
    
    # Checkpoint path for saving during training
    checkpoint_path = CHECKPOINT_DIR / 'training_checkpoint.pt'
    
    # Step 1: Load metadata
    logger.info("\n[Step 1/5] Loading metadata...")
    metadata = load_metadata()
    num_classes = metadata['num_classes']
    
    # Step 2: Create data loaders
    logger.info("\n[Step 2/5] Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders()
    
    # Step 3: Build model
    logger.info("\n[Step 3/5] Building model...")
    model = build_model(num_classes)
    
    # Initialize training variables
    start_epoch = 0
    start_phase = 'train'
    history1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    history2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"\nResuming from checkpoint: {resume_path}")
            # Create dummy optimizer and scheduler for loading
            optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            start_epoch, start_phase, loaded_history, best_val_acc = load_checkpoint(
                model, optimizer, scheduler, resume_path
            )
            
            if start_phase == 'train':
                history1 = loaded_history
            else:
                history1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
                history2 = loaded_history
                # Unfreeze model for fine-tuning phase
                model = unfreeze_model(model, num_layers=20)
            
            logger.info(f"Resumed from epoch {start_epoch}, phase: {start_phase}")
            logger.info(f"Best validation accuracy so far: {best_val_acc:.2f}%")
        else:
            logger.warning(f"Checkpoint not found: {resume_path}. Starting fresh training.")
    
    # Step 4: Train model
    logger.info("\n[Step 4/5] Training model...")
    
    # Phase 1: Train with frozen backbone (if not already completed)
    if start_phase == 'train':
        train_checkpoint_path = CHECKPOINT_DIR / 'phase1_checkpoint.pt'
        model, history1, best_val_acc = train_model(
            model, train_loader, val_loader, args.epochs, LEARNING_RATE,
            start_epoch=start_epoch, existing_history=history1, 
            existing_best_acc=best_val_acc, checkpoint_path=train_checkpoint_path
        )
        start_epoch = 0  # Reset for phase 2
    
    # Phase 2: Fine-tune (if not already completed)
    if start_phase == 'finetune' or (start_phase == 'train' and start_epoch == 0):
        if start_phase == 'train':
            model = unfreeze_model(model, num_layers=20)
        
        finetune_checkpoint_path = CHECKPOINT_DIR / 'phase2_checkpoint.pt'
        model, history2, best_val_acc = train_model(
            model, train_loader, val_loader, args.fine_tune_epochs, FINE_TUNE_LR, 
            fine_tune=True, start_epoch=start_epoch if start_phase == 'finetune' else 0,
            existing_history=history2 if start_phase == 'finetune' else None,
            existing_best_acc=best_val_acc, checkpoint_path=finetune_checkpoint_path
        )
    
    # Combine histories
    history = {
        'train_loss': history1['train_loss'] + history2['train_loss'],
        'train_acc': history1['train_acc'] + history2['train_acc'],
        'val_loss': history1['val_loss'] + history2['val_loss'],
        'val_acc': history1['val_acc'] + history2['val_acc']
    }
    
    # Step 5: Evaluate model
    logger.info("\n[Step 5/5] Evaluating model...")
    class_to_idx = train_loader.dataset.class_to_idx
    metrics, y_pred, y_true = evaluate_model(model, test_loader, class_names)
    
    # Save final model and results
    save_final_model(model, history, metrics, class_names, class_to_idx, best_val_acc)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"\nModel saved to: {MODEL_DIR}")
    logger.info(f"Best model: {MODEL_DIR / 'disease_mobilenet_best.pt'}")
    logger.info(f"Final model: {MODEL_DIR / 'disease_mobilenet_model.pt'}")


if __name__ == '__main__':
    main()
