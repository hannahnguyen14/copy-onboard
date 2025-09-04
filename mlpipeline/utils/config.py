from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for the churn pipeline using Pydantic for validation."""

    dataset_path: Path = Field(..., description="Path to the dataset file")
    target_column: str = Field(..., description="Name of the target column")
    valid_size: float = Field(
        default=0.2, ge=0.0, lt=1.0, description="Validation set size"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    model_class: str = Field(..., description="Model class to use")
    metrics: list[str] = Field(..., description="Metrics to use for evaluation")


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file path."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml

    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}

    return Config.model_validate(data)
