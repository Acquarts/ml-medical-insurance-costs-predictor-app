"""
Training Script for Medical Insurance Cost Prediction Model
This script can be run locally or on Vertex AI Training
Author: Adrián Zambrana
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from google.cloud import storage


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from local path or GCS"""
    if data_path.startswith("gs://"):
        # Load from GCS
        return pd.read_csv(data_path)
    else:
        return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the insurance dataset"""
    # Remove duplicates
    df = df.drop_duplicates()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["sex", "smoker"], drop_first=True)

    # Convert boolean columns to int
    for col in ["sex_male", "smoker_yes"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # One-hot encode region
    region_dummies = pd.get_dummies(df["region"], prefix="region")
    df = pd.concat([df, region_dummies], axis=1)
    df = df.drop(columns=["region"])

    # Separate features and target
    X = df.drop(columns=["charges"])
    y = df["charges"]

    return X, y


def train_model(X_train, y_train, **params):
    """Train the Gradient Boosting model"""
    model = GradientBoostingRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("learning_rate", 0.1),
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate the model and return metrics"""
    y_pred = model.predict(X_test)

    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return metrics


def save_model(model, output_path: str):
    """Save model to local path or GCS"""
    if output_path.startswith("gs://"):
        # Save to GCS
        local_path = "/tmp/model.joblib"
        joblib.dump(model, local_path)

        # Parse GCS path
        path_parts = output_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_path = "/".join(path_parts[1:])

        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)

        print(f"Model saved to {output_path}")
    else:
        # Save locally
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train insurance cost prediction model")

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/insurance.csv",
        help="Path to the training data (local or gs://)"
    )

    # Model arguments
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators for Gradient Boosting"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth of trees"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate"
    )

    # Output arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("AIP_MODEL_DIR", "model"),
        help="Directory to save the model"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Medical Insurance Cost Prediction - Training")
    print("=" * 50)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = load_data(args.data_path)
    print(f"Dataset shape: {df.shape}")

    # Preprocess
    print("\nPreprocessing data...")
    X, y = preprocess_data(df)
    print(f"Features shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train model
    print("\nTraining Gradient Boosting model...")
    model = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  MAE: ${metrics['mae']:,.2f}")
    print(f"  RMSE: ${metrics['rmse']:,.2f}")

    # Feature importance
    print("\nFeature Importance:")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    for feat, imp in importance.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    print(f"\nSaving model to: {model_path}")
    save_model(model, model_path)

    # Save feature names for inference
    feature_names_path = os.path.join(args.model_dir, "feature_names.joblib")
    save_model(list(X.columns), feature_names_path)

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
