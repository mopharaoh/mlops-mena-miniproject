from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_categorical_fill_values(
    X,
    categorical_features: list[str],
) -> dict[str, str]:
    """
    Calculate categorical missing-value replacements
    from the training data only.
    """

    fill_values: dict[str, str] = {}

    for column in categorical_features:
        mode = X[column].mode()

        if mode.empty:
            fill_values[column] = "Missing"
        else:
            fill_values[column] = str(mode.iloc[0])

    return fill_values


def fill_categorical_missing_values(
    X,
    categorical_features: list[str],
    fill_values: dict[str, str],
):
    """
    Fill categorical missing values using statistics
    learned from the training data.
    """

    X = X.copy()

    for column in categorical_features:
        X[column] = X[column].fillna(
            fill_values[column]
        )

    return X


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Categorical missing values are already handled before
    this pipeline is fitted so that the resulting pipeline
    can be exported cleanly to ONNX.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )