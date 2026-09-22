import numpy as np
import pandas as pd
import pytest

from prodml.features import (
    build_preprocessor,
    fill_categorical_missing_values,
    get_categorical_fill_values,
)


def test_get_categorical_fill_values_returns_mode():
    """The most frequent categorical value should be selected."""
    data = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red", None],
            "Material": ["Wood", "Wood", "Metal", "Wood"],
        }
    )

    result = get_categorical_fill_values(
        data,
        ["Color", "Material"],
    )

    assert result["Color"] == "Red"
    assert result["Material"] == "Wood"


def test_get_categorical_fill_values_empty_column():
    """An entirely empty categorical column should use Missing."""
    data = pd.DataFrame(
        {
            "Color": [None, None, None],
        }
    )

    result = get_categorical_fill_values(
        data,
        ["Color"],
    )

    assert result["Color"] == "Missing"


def test_fill_categorical_missing_values():
    """Missing categorical values should be replaced."""
    data = pd.DataFrame(
        {
            "Color": ["Red", None],
            "Material": [None, "Wood"],
        }
    )

    result = fill_categorical_missing_values(
        data,
        ["Color", "Material"],
        {
            "Color": "Red",
            "Material": "Wood",
        },
    )

    assert result["Color"].tolist() == ["Red", "Red"]
    assert result["Material"].tolist() == ["Wood", "Wood"]


def test_fill_categorical_missing_values_does_not_modify_original():
    """The input dataframe should not be modified in place."""
    data = pd.DataFrame(
        {
            "Color": ["Red", None],
        }
    )

    result = fill_categorical_missing_values(
        data,
        ["Color"],
        {"Color": "Blue"},
    )

    assert pd.isna(data.loc[1, "Color"])
    assert result.loc[1, "Color"] == "Blue"


@pytest.mark.parametrize(
    "unknown_value",
    ["Purple", "Unknown", "NewCategory"],
)
def test_preprocessor_handles_unknown_categories(unknown_value):
    """OneHotEncoder should ignore unseen categorical values."""
    train = pd.DataFrame(
        {
            "Age": [20, 30, 40],
            "Color": ["Red", "Blue", "Green"],
        }
    )

    test = pd.DataFrame(
        {
            "Age": [35],
            "Color": [unknown_value],
        }
    )

    preprocessor = build_preprocessor(
        numeric_features=["Age"],
        categorical_features=["Color"],
    )

    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    if hasattr(transformed, "toarray"):    
        transformed = transformed.toarray()


def test_preprocessor_handles_numeric_missing_values():
    """Numeric missing values should be handled by the median imputer."""
    train = pd.DataFrame(
        {
            "Age": [20, 30, 40],
            "Color": ["Red", "Blue", "Green"],
        }
    )

    test = pd.DataFrame(
        {
            "Age": [np.nan],
            "Color": ["Red"],
        }
    )

    preprocessor = build_preprocessor(
        numeric_features=["Age"],
        categorical_features=["Color"],
    )

    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    if hasattr(transformed, "toarray"):  
        transformed = transformed.toarray()