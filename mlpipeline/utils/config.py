from __future__ import annotations

from pathlib import Path
from typing import Optional, Type, overload

import yaml
from pydantic import BaseModel, Field


class PreprocessConfig(BaseModel):
    numeric_keys: list[str] = Field(..., description="numeric columns")
    id_keys: list[str] = Field(..., description="ID column")


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
    preprocess: Optional[Path] = Field(
        default=None, description="Path to preprocess YAML"
    )


@overload
def load_config(config_path: str | Path, model: Type[Config] = ...) -> Config:
    ...


@overload
def load_config(
    config_path: str | Path, model: Type[PreprocessConfig]
) -> PreprocessConfig:
    ...


def load_config(config_path: str | Path, model: Type[BaseModel] = Config) -> BaseModel:
    """Load a YAML file and validate it against the provided Pydantic model."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return model.model_validate(data)
