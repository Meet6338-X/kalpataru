"""
Data Organization Script for Kalpataru.

This script organizes raw datasets from the Dataset folder into the proper
data directory structure (raw, processed, external, analysis).
"""

import os
import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATASET_DIR = BASE_DIR / 'Dataset'
DATA_DIR = BASE_DIR / 'data'

# Target directories
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
EXTERNAL_DIR = DATA_DIR / 'external'
ANALYSIS_DIR = DATA_DIR / 'analysis'


def create_directory_structure():
    """Create the data directory structure."""
    directories = [
        # Raw data directories
        RAW_DIR / 'crop_recommendation',
        RAW_DIR / 'crop_yield',
        RAW_DIR / 'disease_images',
        
        # Processed data directories
        PROCESSED_DIR / 'crop_recommendation',
        PROCESSED_DIR / 'crop_yield',
        PROCESSED_DIR / 'disease_images' / 'train',
        PROCESSED_DIR / 'disease_images' / 'val',
        PROCESSED_DIR / 'disease_images' / 'test',
        
        # External data directories
        EXTERNAL_DIR,
        
        # Analysis directories
        ANALYSIS_DIR / 'crop_recommendation',
        ANALYSIS_DIR / 'crop_yield',
        ANALYSIS_DIR / 'disease_detection',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Directory structure created successfully!")


def organize_crop_recommendation_data():
    """Organize crop recommendation dataset."""
    source_file = SOURCE_DATASET_DIR / 'crop recommendations' / 'Crop_recommendation.csv'
    target_dir = RAW_DIR / 'crop_recommendation'
    target_file = target_dir / 'Crop_recommendation.csv'
    
    if source_file.exists():
        if not target_file.exists():
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied crop recommendation data to {target_file}")
        else:
            logger.info("Crop recommendation data already exists in raw folder")
    else:
        logger.warning(f"Source file not found: {source_file}")


def organize_crop_yield_data():
    """Organize crop yield dataset."""
    source_file = SOURCE_DATASET_DIR / 'crop yield' / 'crop_yield.csv'
    target_dir = RAW_DIR / 'crop_yield'
    target_file = target_dir / 'crop_yield.csv'
    
    if source_file.exists():
        if not target_file.exists():
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied crop yield data to {target_file}")
        else:
            logger.info("Crop yield data already exists in raw folder")
    else:
        logger.warning(f"Source file not found: {source_file}")


def organize_disease_images():
    """
    Organize disease image datasets.
    
    We'll use the augmented dataset as the primary source and create
    symbolic links or copy the directory structure.
    """
    source_dir = SOURCE_DATASET_DIR / 'Plant_leave_diseases_dataset_with_augmentation'
    target_dir = RAW_DIR / 'disease_images' / 'Plant_leave_diseases_augmented'
    
    if source_dir.exists():
        if not target_dir.exists():
            # Copy the entire directory structure
            logger.info("Copying disease image dataset (this may take a while)...")
            shutil.copytree(source_dir, target_dir)
            logger.info(f"Copied disease images to {target_dir}")
        else:
            logger.info("Disease images already exist in raw folder")
    else:
        logger.warning(f"Source directory not found: {source_dir}")
    
    # Also copy the non-augmented version for comparison
    source_dir_no_aug = SOURCE_DATASET_DIR / 'Plant_leave_diseases_dataset_without_augmentation'
    target_dir_no_aug = RAW_DIR / 'disease_images' / 'Plant_leave_diseases_original'
    
    if source_dir_no_aug.exists():
        if not target_dir_no_aug.exists():
            logger.info("Copying original (non-augmented) disease image dataset...")
            shutil.copytree(source_dir_no_aug, target_dir_no_aug)
            logger.info(f"Copied original disease images to {target_dir_no_aug}")
        else:
            logger.info("Original disease images already exist in raw folder")


def create_metadata_files():
    """Create metadata files documenting the data sources."""
    
    # Crop recommendation metadata
    crop_rec_meta = """# Crop Recommendation Dataset

## Source
- Original Location: Dataset/crop recommendations/Crop_recommendation.csv
- Format: CSV

## Description
Dataset for crop recommendation based on soil and climate parameters.

## Expected Columns
- N: Nitrogen content in soil
- P: Phosphorus content in soil
- K: Potassium content in soil
- temperature: Temperature in Celsius
- humidity: Humidity in %
- ph: pH value of soil
- rainfall: Rainfall in mm
- label: Recommended crop (target variable)

## Usage
Used for training the crop recommendation classification model.
"""
    
    meta_file = RAW_DIR / 'crop_recommendation' / 'README.md'
    with open(meta_file, 'w') as f:
        f.write(crop_rec_meta)
    logger.info(f"Created metadata file: {meta_file}")
    
    # Crop yield metadata
    crop_yield_meta = """# Crop Yield Dataset

## Source
- Original Location: Dataset/crop yield/crop_yield.csv
- Format: CSV

## Description
Dataset for predicting crop yield based on various agricultural parameters.

## Usage
Used for training the crop yield regression model.
"""
    
    meta_file = RAW_DIR / 'crop_yield' / 'README.md'
    with open(meta_file, 'w') as f:
        f.write(crop_yield_meta)
    logger.info(f"Created metadata file: {meta_file}")
    
    # Disease images metadata
    disease_meta = """# Plant Disease Image Dataset

## Source
- Original Location: Dataset/Plant_leave_diseases_dataset_with_augmentation
- Format: JPG images organized by class folders

## Description
Dataset for plant disease detection using CNN.

## Classes (39 total)
- Apple: Apple_scab, Black_rot, Cedar_apple_rust, healthy
- Blueberry: healthy
- Cherry: healthy, Powdery_mildew
- Corn: Cercospora_leaf_spot Gray_leaf_spot, Common_rust, healthy, Northern_Leaf_Blight
- Grape: Black_rot, Esca_(Black_Measles), healthy, Leaf_blight_(Isariopsis_Leaf_Spot)
- Orange: Haunglongbing_(Citrus_greening)
- Peach: Bacterial_spot, healthy
- Pepper_bell: Bacterial_spot, healthy
- Potato: Early_blight, healthy, Late_blight
- Raspberry: healthy
- Soybean: healthy
- Squash: Powdery_mildew
- Strawberry: healthy, Leaf_scorch
- Tomato: Bacterial_spot, Early_blight, healthy, Late_blight, Leaf_Mold, 
           Septoria_leaf_spot, Spider_mites, Target_Spot, Tomato_mosaic_virus, 
           Yellow_Leaf_Curl_Virus

## Usage
Used for training the disease detection CNN model.
"""
    
    meta_file = RAW_DIR / 'disease_images' / 'README.md'
    with open(meta_file, 'w') as f:
        f.write(disease_meta)
    logger.info(f"Created metadata file: {meta_file}")


def get_dataset_statistics():
    """Get basic statistics about the organized datasets."""
    stats = {}
    
    # Crop recommendation stats
    crop_rec_file = RAW_DIR / 'crop_recommendation' / 'Crop_recommendation.csv'
    if crop_rec_file.exists():
        stats['crop_recommendation'] = {
            'file_size_kb': crop_rec_file.stat().st_size / 1024,
            'exists': True
        }
    
    # Crop yield stats
    crop_yield_file = RAW_DIR / 'crop_yield' / 'crop_yield.csv'
    if crop_yield_file.exists():
        stats['crop_yield'] = {
            'file_size_kb': crop_yield_file.stat().st_size / 1024,
            'exists': True
        }
    
    # Disease images stats
    disease_dir = RAW_DIR / 'disease_images' / 'Plant_leave_diseases_augmented'
    if disease_dir.exists():
        class_count = len([d for d in disease_dir.iterdir() if d.is_dir()])
        total_images = sum(
            len(list(d.iterdir())) for d in disease_dir.iterdir() if d.is_dir()
        )
        stats['disease_images'] = {
            'classes': class_count,
            'total_images': total_images,
            'exists': True
        }
    
    return stats


def main():
    """Main function to organize all data."""
    logger.info("=" * 60)
    logger.info("Starting data organization for Kalpataru")
    logger.info("=" * 60)
    
    # Step 1: Create directory structure
    logger.info("\n[Step 1/5] Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Organize crop recommendation data
    logger.info("\n[Step 2/5] Organizing crop recommendation data...")
    organize_crop_recommendation_data()
    
    # Step 3: Organize crop yield data
    logger.info("\n[Step 3/5] Organizing crop yield data...")
    organize_crop_yield_data()
    
    # Step 4: Organize disease images
    logger.info("\n[Step 4/5] Organizing disease images...")
    organize_disease_images()
    
    # Step 5: Create metadata files
    logger.info("\n[Step 5/5] Creating metadata files...")
    create_metadata_files()
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("Data organization complete!")
    logger.info("=" * 60)
    
    stats = get_dataset_statistics()
    logger.info("\nDataset Statistics:")
    for dataset, info in stats.items():
        logger.info(f"  {dataset}: {info}")
    
    logger.info("\nNext steps:")
    logger.info("  1. Run preprocess_crop_recommendation.py")
    logger.info("  2. Run preprocess_crop_yield.py")
    logger.info("  3. Run preprocess_disease_images.py")
    logger.info("  4. Run training scripts")


if __name__ == '__main__':
    main()
