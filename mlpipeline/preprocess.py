# mlpipeline/preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from mlpipeline.cleaning import (
    coerce_numeric_columns,
    drop_id_duplicates_then_full,
    ensure_target_binary,
    normalize_basic_categoricals,
    normalize_column_names,
    remove_garbage_tokens,
    standardize_na_tokens,
    strip_text_whitespace,
)
from mlpipeline.utils.config import Config, PreprocessConfig, load_config
from mlpipeline.utils.logger import logger


class Preprocessor:
    """Preprocessor class to load and clean raw datasets."""

    def __init__(self, config: Config) -> None:
        self.config: Final[Config] = config

        # Load preprocess.yaml if provided
        if self.config.preprocess:
            pp: PreprocessConfig = load_config(self.config.preprocess, PreprocessConfig)
            self._numeric_keys = pp.numeric_keys
            self._id_keys = pp.id_keys
        else:
            self._numeric_keys = []
            self._id_keys = []

    def run(self) -> pd.DataFrame:
        """Run the full preprocessing: load then clean."""
        logger.info("Preprocessor.run() [load -> clean]")
        return self.clean(self.load())

    def load(self) -> pd.DataFrame:
        """Load dataset from CSV path in config."""
        path = Path(self.config.dataset_path)
        logger.info(f"Loading dataset from: {path.as_posix()}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")
        return pd.read_csv(path).copy(deep=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by applying a series of safe transformations."""
        dfx = df.copy(deep=True)
        dfx = normalize_column_names(dfx)
        dfx = strip_text_whitespace(dfx)
        dfx = remove_garbage_tokens(dfx)
        dfx = standardize_na_tokens(dfx)
        dfx = normalize_basic_categoricals(dfx)
        dfx = coerce_numeric_columns(dfx, self._numeric_keys)
        # ép ID thành string & bỏ .0
        for id_col in self._id_keys:
            if id_col in dfx.columns:
                dfx[id_col] = (
                    dfx[id_col].astype("string").str.replace(r"\.0$", "", regex=True)
                )
        tgt = self.config.target_column  # Chỉ chuẩn hoá khi có target
        if tgt in dfx.columns or "attrition_flag" in dfx.columns:
            dfx = ensure_target_binary(dfx, target_col=tgt)
        dfx = drop_id_duplicates_then_full(dfx, id_keys=self._id_keys)
        return dfx

