from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration."""

    data_path: Path = Path("data/raw/train.csv")

    model_path: Path = Path(
        "models/model.pkl"
    )

    onnx_model_path: Path = Path(
        "models/model.onnx"
    )

    target_column: str = "SalePrice"

    validation_size: float = 0.2

    random_state: int = 42


settings = Settings()