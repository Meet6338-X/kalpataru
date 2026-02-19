"""
Master Pipeline Script for Kalpataru.

This script runs the complete data organization, preprocessing, and training pipeline.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parent.parent / 'logs' / 'pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / 'scripts'


def run_step(step_name, script_name):
    """Run a single pipeline step."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running: {step_name}")
    logger.info(f"{'=' * 60}")
    
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Run the script
    exit_code = os.system(f'python "{script_path}"')
    
    if exit_code == 0:
        logger.info(f"✓ {step_name} completed successfully")
        return True
    else:
        logger.error(f"✗ {step_name} failed with exit code {exit_code}")
        return False


def run_full_pipeline(skip_organize=False, skip_preprocess=False, skip_train=False):
    """Run the complete pipeline."""
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("KALPATARU - Full Training Pipeline")
    logger.info(f"Started at: {start_time}")
    logger.info("=" * 60)
    
    results = {}
    
    # Phase 1: Data Organization
    if not skip_organize:
        logger.info("\n### PHASE 1: DATA ORGANIZATION ###")
        results['organize_data'] = run_step(
            "Data Organization",
            "organize_data.py"
        )
    else:
        logger.info("\n### PHASE 1: DATA ORGANIZATION (SKIPPED) ###")
        results['organize_data'] = True
    
    # Phase 2: Preprocessing
    if not skip_preprocess:
        logger.info("\n### PHASE 2: DATA PREPROCESSING ###")
        
        results['preprocess_crop_recommendation'] = run_step(
            "Crop Recommendation Preprocessing",
            "preprocess_crop_recommendation.py"
        )
        
        results['preprocess_crop_yield'] = run_step(
            "Crop Yield Preprocessing",
            "preprocess_crop_yield.py"
        )
        
        results['preprocess_disease_images'] = run_step(
            "Disease Images Preprocessing",
            "preprocess_disease_images.py"
        )
    else:
        logger.info("\n### PHASE 2: DATA PREPROCESSING (SKIPPED) ###")
        results['preprocess_crop_recommendation'] = True
        results['preprocess_crop_yield'] = True
        results['preprocess_disease_images'] = True
    
    # Phase 3: Training
    if not skip_train:
        logger.info("\n### PHASE 3: MODEL TRAINING ###")
        
        results['train_crop_recommendation'] = run_step(
            "Crop Recommendation Model Training",
            "train_crop_recommendation.py"
        )
        
        results['train_crop_yield'] = run_step(
            "Crop Yield Model Training",
            "train_crop_yield.py"
        )
        
        results['train_disease_detection'] = run_step(
            "Disease Detection Model Training",
            "train_disease_detection.py"
        )
    else:
        logger.info("\n### PHASE 3: MODEL TRAINING (SKIPPED) ###")
        results['train_crop_recommendation'] = True
        results['train_crop_yield'] = True
        results['train_disease_detection'] = True
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    for step, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {step}: {status}")
    
    logger.info(f"\nTotal duration: {duration}")
    logger.info(f"Completed at: {end_time}")
    
    # Check if all steps succeeded
    all_success = all(results.values())
    
    if all_success:
        logger.info("\n🎉 All pipeline steps completed successfully!")
    else:
        logger.warning("\n⚠️ Some pipeline steps failed. Check the logs for details.")
    
    return all_success


def run_quick_training():
    """Run only the training phase (assumes data is already preprocessed)."""
    return run_full_pipeline(skip_organize=True, skip_preprocess=True, skip_train=False)


def run_preprocessing_only():
    """Run only the preprocessing phase (assumes data is already organized)."""
    return run_full_pipeline(skip_organize=True, skip_preprocess=False, skip_train=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Kalpataru Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --skip-organize    # Skip data organization
  python run_pipeline.py --skip-preprocess  # Skip preprocessing
  python run_pipeline.py --skip-train       # Skip training
  python run_pipeline.py --train-only       # Only run training
  python run_pipeline.py --preprocess-only  # Only run preprocessing
        """
    )
    
    parser.add_argument(
        '--skip-organize',
        action='store_true',
        help='Skip data organization step'
    )
    
    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing steps'
    )
    
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training steps'
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Run only training (skip organize and preprocess)'
    )
    
    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        help='Run only preprocessing (skip organize and train)'
    )
    
    args = parser.parse_args()
    
    # Handle shortcut options
    if args.train_only:
        success = run_quick_training()
    elif args.preprocess_only:
        success = run_preprocessing_only()
    else:
        success = run_full_pipeline(
            skip_organize=args.skip_organize,
            skip_preprocess=args.skip_preprocess,
            skip_train=args.skip_train
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()