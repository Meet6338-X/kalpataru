"""
Preprocessing Script for Plant Disease Image Dataset.

This script preprocesses the raw disease images and prepares
them for training the CNN-based disease detection model.
"""

import os
import shutil
import logging
import json
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw' / 'disease_images'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / 'disease_images'

# Image processing settings
IMAGE_SIZE = (224, 224)  # Standard size for CNN models
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


def get_all_classes(source_dir):
    """Get all disease classes from the source directory."""
    classes = []
    for item in source_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            classes.append(item.name)
    return sorted(classes)


def count_images_per_class(source_dir, classes):
    """Count images per class."""
    class_counts = {}
    for class_name in classes:
        class_dir = source_dir / class_name
        if class_dir.exists():
            count = len([
                f for f in class_dir.iterdir() 
                if f.suffix.lower() in VALID_EXTENSIONS
            ])
            class_counts[class_name] = count
    return class_counts


def validate_image(image_path):
    """Validate that an image can be opened and processed."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return False


def process_image(image_path, target_size=IMAGE_SIZE):
    """Process a single image: resize and convert to RGB."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            return img
    except Exception as e:
        logger.warning(f"Error processing {image_path}: {e}")
        return None


def split_data(class_counts, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT):
    """
    Create train/val/test split indices for each class.
    Returns dict with class_name -> {'train': [...], 'val': [...], 'test': [...]}
    """
    splits = {}
    
    for class_name, count in class_counts.items():
        indices = list(range(count))
        random.shuffle(indices)
        
        train_end = int(count * train_split)
        val_end = train_end + int(count * val_split)
        
        splits[class_name] = {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:]
        }
    
    return splits


def create_output_directories(classes):
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            output_dir = PROCESSED_DIR / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directories in {PROCESSED_DIR}")


def process_and_copy_images(source_dir, classes, splits):
    """Process images and copy to appropriate split directories."""
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'invalid': 0})
    
    for class_name in classes:
        logger.info(f"Processing class: {class_name}")
        
        class_dir = source_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        # Get all valid image files
        image_files = sorted([
            f for f in class_dir.iterdir() 
            if f.suffix.lower() in VALID_EXTENSIONS
        ])
        
        for idx, image_file in enumerate(image_files):
            # Determine which split this image belongs to
            if idx in splits[class_name]['train']:
                split = 'train'
            elif idx in splits[class_name]['val']:
                split = 'val'
            elif idx in splits[class_name]['test']:
                split = 'test'
            else:
                # This shouldn't happen, but handle gracefully
                stats[class_name]['invalid'] += 1
                continue
            
            # Process image
            processed_img = process_image(image_file)
            
            if processed_img is not None:
                # Save to output directory
                output_path = PROCESSED_DIR / split / class_name / image_file.name
                processed_img.save(output_path, 'JPEG', quality=95)
                stats[class_name][split] += 1
            else:
                stats[class_name]['invalid'] += 1
        
        logger.info(f"  Train: {stats[class_name]['train']}, "
                   f"Val: {stats[class_name]['val']}, "
                   f"Test: {stats[class_name]['test']}, "
                   f"Invalid: {stats[class_name]['invalid']}")
    
    return dict(stats)


def create_class_mapping(classes):
    """Create class name to index mapping."""
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    return class_to_idx, idx_to_class


def save_metadata(classes, class_counts, stats, class_to_idx):
    """Save metadata about the processed dataset."""
    metadata = {
        'num_classes': len(classes),
        'classes': classes,
        'class_to_idx': class_to_idx,
        'image_size': IMAGE_SIZE,
        'splits': {
            'train': TRAIN_SPLIT,
            'val': VAL_SPLIT,
            'test': TEST_SPLIT
        },
        'original_counts': class_counts,
        'processed_counts': stats
    }
    
    # Calculate totals
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    
    metadata['totals'] = {
        'train': total_train,
        'val': total_val,
        'test': total_test,
        'all': total_train + total_val + total_test
    }
    
    # Save metadata
    metadata_path = PROCESSED_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Also save as Python constants for easy import
    constants_path = PROCESSED_DIR / 'class_constants.py'
    with open(constants_path, 'w') as f:
        f.write('"""Auto-generated class constants for disease detection."""\n\n')
        f.write(f'NUM_CLASSES = {len(classes)}\n\n')
        f.write('CLASS_NAMES = [\n')
        for class_name in classes:
            f.write(f'    "{class_name}",\n')
        f.write(']\n\n')
        f.write('CLASS_TO_IDX = {\n')
        for class_name, idx in class_to_idx.items():
            f.write(f'    "{class_name}": {idx},\n')
        f.write('}\n')
    
    logger.info(f"Saved class constants to {constants_path}")
    
    return metadata


def print_summary(metadata):
    """Print a summary of the processed dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Summary")
    logger.info("=" * 60)
    
    logger.info(f"\nTotal classes: {metadata['num_classes']}")
    logger.info(f"Image size: {metadata['image_size']}")
    
    logger.info("\nSplit distribution:")
    totals = metadata['totals']
    logger.info(f"  Training:   {totals['train']:>6} images ({TRAIN_SPLIT*100:.0f}%)")
    logger.info(f"  Validation: {totals['val']:>6} images ({VAL_SPLIT*100:.0f}%)")
    logger.info(f"  Test:       {totals['test']:>6} images ({TEST_SPLIT*100:.0f}%)")
    logger.info(f"  Total:      {totals['all']:>6} images")
    
    logger.info("\nClasses:")
    for class_name in metadata['classes'][:10]:  # Show first 10
        count = metadata['original_counts'].get(class_name, 0)
        logger.info(f"  - {class_name}: {count} images")
    
    if len(metadata['classes']) > 10:
        logger.info(f"  ... and {len(metadata['classes']) - 10} more classes")


def filter_classes(classes, filter_keywords=None):
    """
    Filter classes based on keywords.
    Useful for training on specific crops only.
    """
    if filter_keywords is None:
        return classes
    
    filtered = []
    for class_name in classes:
        for keyword in filter_keywords:
            if keyword.lower() in class_name.lower():
                filtered.append(class_name)
                break
    
    return filtered


def main():
    """Main preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Plant Disease Image Preprocessing")
    logger.info("=" * 60)
    
    # Determine source directory
    source_dir = RAW_DIR / 'Plant_leave_diseases_augmented'
    if not source_dir.exists():
        source_dir = RAW_DIR / 'Plant_leave_diseases_original'
    
    if not source_dir.exists():
        logger.error(f"Source directory not found. Please run organize_data.py first.")
        return
    
    logger.info(f"\nUsing source directory: {source_dir}")
    
    # Step 1: Get all classes
    logger.info("\n[Step 1/6] Getting all classes...")
    classes = get_all_classes(source_dir)
    logger.info(f"Found {len(classes)} classes")
    
    # Optional: Filter to specific crops
    # Uncomment the following line to filter to Tomato only:
    # classes = filter_classes(classes, ['Tomato'])
    
    # Step 2: Count images per class
    logger.info("\n[Step 2/6] Counting images per class...")
    class_counts = count_images_per_class(source_dir, classes)
    total_images = sum(class_counts.values())
    logger.info(f"Total images: {total_images}")
    
    # Step 3: Create splits
    logger.info("\n[Step 3/6] Creating train/val/test splits...")
    random.seed(42)  # For reproducibility
    splits = split_data(class_counts)
    
    # Step 4: Create output directories
    logger.info("\n[Step 4/6] Creating output directories...")
    create_output_directories(classes)
    
    # Step 5: Process and copy images
    logger.info("\n[Step 5/6] Processing and copying images...")
    stats = process_and_copy_images(source_dir, classes, splits)
    
    # Step 6: Save metadata
    logger.info("\n[Step 6/6] Saving metadata...")
    class_to_idx, idx_to_class = create_class_mapping(classes)
    metadata = save_metadata(classes, class_counts, stats, class_to_idx)
    
    # Print summary
    print_summary(metadata)
    
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    logger.info(f"\nProcessed data saved to: {PROCESSED_DIR}")
    logger.info("\nNext step: Run train_disease_detection.py")


if __name__ == '__main__':
    main()