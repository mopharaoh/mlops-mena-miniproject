from pathlib import Path

import joblib
import pandas as pd
import pytest

from prodml.predict import (
    FeatureValidationError,
    HousePricePredictor,
)


def test_predictor_feature_names(trained_model):
    """The predictor should expose the features expected by the model."""
    feature_names = trained_model.feature_names

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert "MSSubClass" in feature_names


def test_predictor_categorical_features(trained_model):
    """Categorical feature names should come from fill-value configuration."""
    categorical_features = trained_model.categorical_features

    assert isinstance(categorical_features, list)
    assert len(categorical_features) > 0


def test_predict_one(trained_model, sample_features):
    """A valid feature dictionary should produce one numeric prediction."""
    prediction = trained_model.predict_one(sample_features)

    assert isinstance(prediction, float)
    assert prediction > 0


def test_predict_batch(trained_model, sample_features):
    """Batch prediction should return one prediction per input."""
    features = [
        sample_features,
        sample_features,
    ]

    predictions = trained_model.predict_batch(features)

    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert all(isinstance(value, float) for value in predictions)


def test_predict_batch_empty_input(trained_model):
    """Empty batch input should raise a clear ValueError."""
    with pytest.raises(
        ValueError,
        match="At least one feature set is required",
    ):
        trained_model.predict_batch([])


def test_unknown_feature_raises_error(
    trained_model,
    sample_features,
):
    """Unknown feature names should be rejected."""
    invalid_features = {
        **sample_features,
        "NotARealFeature": 123,
    }

    with pytest.raises(
        FeatureValidationError,
        match="Unknown features",
    ):
        trained_model.predict_one(invalid_features)


def test_multiple_unknown_features_are_reported(
    trained_model,
    sample_features,
):
    """Multiple unknown feature names should be included in the error."""
    invalid_features = {
        **sample_features,
        "UnknownA": 1,
        "UnknownB": 2,
    }

    with pytest.raises(
        FeatureValidationError,
        match="UnknownA",
    ):
        trained_model.predict_one(invalid_features)


def test_prepare_for_onnx(trained_model, sample_features):
    """ONNX preparation should preserve the expected feature order."""
    dataframe = pd.DataFrame([sample_features])

    prepared = trained_model.prepare_for_onnx(dataframe)

    assert list(prepared.columns) == trained_model.feature_names
    assert prepared.shape[0] == 1


def test_predictor_load_missing_model(tmp_path):
    """Loading a missing model artifact should raise FileNotFoundError."""
    missing_path = tmp_path / "missing.pkl"

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        HousePricePredictor.load(missing_path)


def test_predictor_load_invalid_artifact(tmp_path):
    """An artifact containing the wrong object type should be rejected."""
    invalid_path = tmp_path / "invalid.pkl"

    joblib.dump({"not": "a predictor"}, invalid_path)

    with pytest.raises(
        TypeError,
        match="HousePricePredictor",
    ):
        HousePricePredictor.load(invalid_path)


def test_feature_names_without_model_metadata(
    trained_model,
):
    """feature_names should fail if the model lacks feature_names_in_."""
    class ModelWithoutFeatureNames:
        pass

    original_model = trained_model.model

    try:
        trained_model.model = ModelWithoutFeatureNames()

        with pytest.raises(
            AttributeError,
            match="feature_names_in_",
        ):
            _ = trained_model.feature_names
    finally:
        trained_model.model = original_model