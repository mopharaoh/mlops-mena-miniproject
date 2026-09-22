from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: Path) -> pd.DataFrame:
    """Load the House Prices dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Id is an identifier, not a predictive feature.
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    return X, y


def train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split features and target into training and validation sets."""

    return train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
    )