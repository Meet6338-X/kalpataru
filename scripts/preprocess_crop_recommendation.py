"""
Preprocessing Script for Crop Recommendation Dataset.

This script preprocesses the raw crop recommendation data and prepares
it for training the crop recommendation model.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw' / 'crop_recommendation'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / 'crop_recommendation'


def load_raw_data():
    """Load the raw crop recommendation data."""
    file_path = RAW_DIR / 'Crop_recommendation.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded raw data with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def explore_data(df):
    """Perform basic data exploration."""
    logger.info("\n" + "=" * 50)
    logger.info("Data Exploration")
    logger.info("=" * 50)
    
    # Basic info
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"\nColumn types:\n{df.dtypes}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        logger.info("\nNo missing values found!")
    
    # Statistical summary
    logger.info(f"\nStatistical summary:\n{df.describe()}")
    
    # Target distribution
    if 'label' in df.columns:
        logger.info(f"\nTarget distribution:\n{df['label'].value_counts()}")
    
    return df


def clean_data(df):
    """Clean the data by handling missing values and outliers."""
    df = df.copy()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"Filled missing values in {col} with median")
    
    # Handle categorical missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info(f"Filled missing values in {col} with mode")
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        logger.info(f"Removed {duplicates} duplicate rows")
    
    return df


def engineer_features(df):
    """Create new features from existing ones."""
    df = df.copy()
    
    # NPK ratio features
    if all(col in df.columns for col in ['N', 'P', 'K']):
        df['N_P_ratio'] = df['N'] / (df['P'] + 1)
        df['N_K_ratio'] = df['N'] / (df['K'] + 1)
        df['P_K_ratio'] = df['P'] / (df['K'] + 1)
        df['NPK_sum'] = df['N'] + df['P'] + df['K']
        logger.info("Created NPK ratio features")
    
    # Temperature-humidity interaction
    if all(col in df.columns for col in ['temperature', 'humidity']):
        df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
        logger.info("Created temperature-humidity index")
    
    # pH categories
    if 'ph' in df.columns:
        df['ph_category'] = pd.cut(
            df['ph'], 
            bins=[0, 5.5, 6.5, 7.5, 8.5, 14],
            labels=['very_acidic', 'acidic', 'neutral', 'alkaline', 'very_alkaline']
        )
        logger.info("Created pH category feature")
    
    # Rainfall categories
    if 'rainfall' in df.columns:
        df['rainfall_category'] = pd.cut(
            df['rainfall'],
            bins=[0, 400, 800, 1200, 2000, float('inf')],
            labels=['very_low', 'low', 'moderate', 'high', 'very_high']
        )
        logger.info("Created rainfall category feature")
    
    return df


def encode_features(df):
    """Encode categorical features and target."""
    df = df.copy()
    encoders = {}
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col != 'label':  # Don't encode target yet
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"Encoded feature: {col}")
    
    # Encode target variable
    if 'label' in df.columns:
        le_target = LabelEncoder()
        df['label_encoded'] = le_target.fit_transform(df['label'])
        encoders['label'] = le_target
        logger.info(f"Encoded target variable with {len(le_target.classes_)} classes")
        logger.info(f"Classes: {list(le_target.classes_)}")
    
    return df, encoders


def prepare_features_target(df):
    """Prepare feature matrix and target vector."""
    # Define feature columns (exclude original categorical and target)
    feature_cols = [
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
        'N_P_ratio', 'N_K_ratio', 'P_K_ratio', 'NPK_sum',
        'temp_humidity_index'
    ]
    
    # Add encoded categorical features if they exist
    for col in df.columns:
        if col.endswith('_encoded') and col != 'label_encoded':
            feature_cols.append(col)
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].values
    y = df['label_encoded'].values if 'label_encoded' in df.columns else None
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features used: {feature_cols}")
    
    return X, y, feature_cols


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, feature_cols, encoders, scaler):
    """Save processed data to files."""
    # Create directory if needed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(PROCESSED_DIR / 'X_train.npy', X_train)
    np.save(PROCESSED_DIR / 'X_test.npy', X_test)
    np.save(PROCESSED_DIR / 'y_train.npy', y_train)
    np.save(PROCESSED_DIR / 'y_test.npy', y_test)
    
    logger.info("Saved feature and target arrays")
    
    # Save feature columns
    with open(PROCESSED_DIR / 'feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save encoders
    with open(PROCESSED_DIR / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save scaler
    with open(PROCESSED_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("Saved encoders and scaler")
    
    # Save processed dataframe with all features
    # This is useful for analysis
    logger.info(f"Processed data saved to {PROCESSED_DIR}")


def create_data_report(df, encoders):
    """Create a data processing report."""
    report = {
        'original_shape': df.shape,
        'features_created': [col for col in df.columns if col not in 
                            ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']],
        'target_classes': list(encoders.get('label', []).classes_) if 'label' in encoders else [],
        'numeric_features': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
        'engineered_features': ['N_P_ratio', 'N_K_ratio', 'P_K_ratio', 'NPK_sum', 'temp_humidity_index']
    }
    
    import json
    with open(PROCESSED_DIR / 'processing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Created processing report")
    
    return report


def main():
    """Main preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Crop Recommendation Data Preprocessing")
    logger.info("=" * 60)
    
    # Step 1: Load raw data
    logger.info("\n[Step 1/7] Loading raw data...")
    df = load_raw_data()
    
    # Step 2: Explore data
    logger.info("\n[Step 2/7] Exploring data...")
    df = explore_data(df)
    
    # Step 3: Clean data
    logger.info("\n[Step 3/7] Cleaning data...")
    df = clean_data(df)
    
    # Step 4: Engineer features
    logger.info("\n[Step 4/7] Engineering features...")
    df = engineer_features(df)
    
    # Step 5: Encode features
    logger.info("\n[Step 5/7] Encoding features...")
    df, encoders = encode_features(df)
    
    # Step 6: Prepare features and target
    logger.info("\n[Step 6/7] Preparing features and target...")
    X, y, feature_cols = prepare_features_target(df)
    
    # Step 7: Split and scale
    logger.info("\n[Step 7/7] Splitting and scaling data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save processed data
    logger.info("\nSaving processed data...")
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, encoders, scaler)
    
    # Create report
    create_data_report(df, encoders)
    
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    logger.info(f"\nProcessed files saved to: {PROCESSED_DIR}")
    logger.info("\nNext step: Run train_crop_recommendation.py")


if __name__ == '__main__':
    main()