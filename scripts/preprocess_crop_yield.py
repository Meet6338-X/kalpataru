"""
Preprocessing Script for Crop Yield Dataset.

This script preprocesses the raw crop yield data and prepares
it for training the yield prediction model.
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
RAW_DIR = BASE_DIR / 'data' / 'raw' / 'crop_yield'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / 'crop_yield'


def load_raw_data():
    """Load the raw crop yield data."""
    file_path = RAW_DIR / 'crop_yield.csv'
    
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
    
    # Unique values in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        logger.info(f"\nUnique values in {col}: {df[col].nunique()}")
        if df[col].nunique() <= 20:
            logger.info(f"  Values: {list(df[col].unique())}")
    
    return df


def clean_data(df):
    """Clean the data by handling missing values and outliers."""
    df = df.copy()
    
    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"Filled missing values in {col} with median")
    
    # Handle missing values for categorical columns
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
    
    # Handle outliers in yield (target) - cap at 99th percentile
    if 'Yield' in df.columns:
        upper_cap = df['Yield'].quantile(0.99)
        lower_cap = df['Yield'].quantile(0.01)
        df['Yield'] = df['Yield'].clip(lower=lower_cap, upper=upper_cap)
        logger.info(f"Capped yield outliers at {lower_cap:.2f} and {upper_cap:.2f}")
    
    return df


def engineer_features(df):
    """Create new features from existing ones."""
    df = df.copy()
    
    # Identify potential columns based on common yield dataset patterns
    # These will be adapted based on actual column names
    
    # If we have area and production, calculate yield if not present
    if 'Area' in df.columns and 'Production' in df.columns and 'Yield' not in df.columns:
        df['Yield'] = df['Production'] / (df['Area'] + 1)  # Avoid division by zero
        logger.info("Calculated yield from production and area")
    
    # Create area categories
    if 'Area' in df.columns:
        df['Area_Category'] = pd.cut(
            df['Area'],
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )
        logger.info("Created area category feature")
    
    # Create season encoding if season column exists
    if 'Season' in df.columns:
        # Map seasons to numeric
        season_map = {
            'Kharif': 1, 'Rabi': 2, 'Zaid': 3, 'Whole Year': 4,
            'Kharif     ': 1, 'Rabi     ': 2, 'Zaid     ': 3,  # Handle trailing spaces
            'Autumn': 1, 'Winter': 2, 'Summer': 3, 'Monsoon': 1
        }
        df['Season_Encoded'] = df['Season'].map(season_map).fillna(0)
        logger.info("Created season encoded feature")
    
    # Create crop year features if Crop_Year exists
    if 'Crop_Year' in df.columns:
        df['Year_Since_2000'] = df['Crop_Year'] - 2000
        logger.info("Created year since 2000 feature")
    
    # Log transform area if it has wide range
    if 'Area' in df.columns:
        df['Log_Area'] = np.log1p(df['Area'])
        logger.info("Created log area feature")
    
    return df


def encode_features(df):
    """Encode categorical features."""
    df = df.copy()
    encoders = {}
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col != 'Yield':  # Don't encode target
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"Encoded feature: {col} ({len(le.classes_)} unique values)")
    
    return df, encoders


def prepare_features_target(df, target_col='Yield'):
    """Prepare feature matrix and target vector."""
    # Identify feature columns
    # Exclude original categorical columns, target, and derived columns
    exclude_cols = [target_col]
    
    # Add original categorical columns to exclude
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    exclude_cols.extend(categorical_cols)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to numeric columns only
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    feature_cols = list(numeric_features)
    
    X = df[feature_cols].values
    
    if target_col in df.columns:
        y = df[target_col].values
    else:
        y = None
        logger.warning(f"Target column '{target_col}' not found!")
    
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
        X, y, test_size=test_size, random_state=random_state
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
    logger.info(f"Processed data saved to {PROCESSED_DIR}")


def create_data_report(df, encoders, feature_cols):
    """Create a data processing report."""
    report = {
        'original_shape': df.shape,
        'features_used': feature_cols,
        'target_column': 'Yield',
        'categorical_encodings': list(encoders.keys()),
        'numeric_stats': {
            'yield_mean': float(df['Yield'].mean()) if 'Yield' in df.columns else None,
            'yield_std': float(df['Yield'].std()) if 'Yield' in df.columns else None,
            'yield_min': float(df['Yield'].min()) if 'Yield' in df.columns else None,
            'yield_max': float(df['Yield'].max()) if 'Yield' in df.columns else None,
        }
    }
    
    import json
    with open(PROCESSED_DIR / 'processing_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("Created processing report")
    
    return report


def main():
    """Main preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Crop Yield Data Preprocessing")
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
    
    if y is None:
        logger.error("Target variable not found. Cannot proceed.")
        return
    
    # Step 7: Split and scale
    logger.info("\n[Step 7/7] Splitting and scaling data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save processed data
    logger.info("\nSaving processed data...")
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, encoders, scaler)
    
    # Create report
    create_data_report(df, encoders, feature_cols)
    
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    logger.info(f"\nProcessed files saved to: {PROCESSED_DIR}")
    logger.info("\nNext step: Run train_crop_yield.py")


if __name__ == '__main__':
    main()