from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: Path) -> pd.DataFrame:
    """Load the House Prices dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    return X, y


def fill_categorical_missing_values(
    X: pd.DataFrame,
    categorical_features: list[str],
    fill_values: dict[str, str],
) -> pd.DataFrame:
    """Fill missing categorical values using training-set values."""

    X = X.copy()

    for column in categorical_features:
        X[column] = X[column].fillna(fill_values[column])

    return X


def get_categorical_fill_values(
    X: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, str]:
    """Get the most frequent value for each categorical feature."""

    fill_values: dict[str, str] = {}

    for column in categorical_features:
        mode = X[column].mode()

        if mode.empty:
            fill_values[column] = "Missing"
        else:
            fill_values[column] = str(mode.iloc[0])

    return fill_values


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
    """Split the dataset into training and validation sets."""

    return train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
    )