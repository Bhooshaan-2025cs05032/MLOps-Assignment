import logging
import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data():
    logger.info("Fetching Heart Disease dataset from UCI ML Repository (id=45)...")
    heart_disease = fetch_ucirepo(id=45)

    columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)
    df.columns = columns
    logger.info("Dataset loaded successfully. Shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning...")

    missing_before = df.isin(["?"]).sum().sum()
    df.replace("?", np.nan, inplace=True)
    logger.info("Replaced %d '?' placeholder(s) with NaN.", missing_before)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.info("Null values detected before imputation:\n%s", cols_with_nulls.to_string())
    else:
        logger.info("No null values detected after '?' replacement.")

    df.fillna(df.median(), inplace=True)
    logger.info("Filled missing values with column medians.")

    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    class_dist = df["target"].value_counts().to_dict()
    logger.info("Target binarized. Class distribution: %s", class_dist)

    logger.info("Data cleaning complete. Final shape: %s", df.shape)
    return df


def save_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned data saved to '%s'.", output_path)


def run_pipeline(output_path="../data/processed/heart_clean.csv"):
    logger.info("=== Data processing pipeline started ===")
    df = load_data()
    df = clean_data(df)
    save_data(df, output_path)
    logger.info("=== Data processing pipeline finished ===")


if __name__ == "__main__":
    run_pipeline()