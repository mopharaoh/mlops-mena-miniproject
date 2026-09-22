from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and training configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_path: Path = Path(
        "data/raw/train.csv"
    )

    model_path: Path = Path(
        "models/model.pkl"
    )

    onnx_model_path: Path = Path(
        "models/model.onnx"
    )

    target_column: str = "SalePrice"

    validation_size: float = 0.2

    random_state: int = 42

    model_version: str = "1.0.0"

    api_host: str = "127.0.0.1"

    api_port: int = 8000


settings = Settings()