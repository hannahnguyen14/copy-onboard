# mlpipeline/cleaning.py
from __future__ import annotations
import re
from typing import List, Set
import numpy as np
import pandas as pd

_GARBAGE_RE = re.compile(r"(@@@|\$\$\$|###)")
_DEFAULT_NA_TOKENS: Set[str] = {"", "na", "n/a", "null", "none", "nan", "unknown"}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names to snake_case."""
    def norm(c: str) -> str:
        nc = c.strip().lower()
        nc = re.sub(r"[^\w]+", "_", nc)
        return re.sub(r"_+", "_", nc).strip("_")
    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out


def strip_text_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace and normalize spaces in text columns."""
    out = df.copy()
    for c in out.select_dtypes(include=["object", "string", "category"]).columns:
        out[c] = (
            out[c].astype("string").str.replace(r"\s+", " ", regex=True).str.strip())
    return out


def remove_garbage_tokens(
    df: pd.DataFrame, pattern: re.Pattern[str] = _GARBAGE_RE
) -> pd.DataFrame:
    """Remove garbage tokens like @@@, $$$, ### from string columns."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]):
            out[c] = (
                out[c].astype("string").str.replace(pattern, "", regex=True).str.strip()
            )
    return out


def standardize_na_tokens(
    df: pd.DataFrame, na_tokens: Set[str] | None = None
) -> pd.DataFrame:
    """Standardize common NA tokens (e.g. 'null', 'unknown') to np.nan."""
    out = df.copy()
    na_tokens = na_tokens or _DEFAULT_NA_TOKENS
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]):
            s = out[c].astype("string")
            out[c] = s.where(~s.str.lower().isin(na_tokens), other=np.nan)
    return out


def normalize_basic_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize education, marital, income, and card categories to title-case."""
    out = df.copy()

    def tidy(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    mapping = {
        "education_level": None,
        "marital_status": None,
        "income_category": {"$40k": "$40K", "$80k": "$80K", "$120k": "$120K"},
        "card_category": None,
    }
    for col, repl in mapping.items():
        if col in out.columns:
            s = tidy(out[col]).str.title()
            if repl:
                for k, v in repl.items():
                    s = s.str.replace(k, v, regex=False)
            out[col] = s
    return out


def coerce_numeric_columns(
    df: pd.DataFrame, preferred_numeric: List[str]
) -> pd.DataFrame:
    """Coerce numeric-like columns to floats, removing '_err' suffix if present."""

    out = df.copy()

    def coerce_one(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype("string")
            .str.replace(r"_err\b", "", regex=True)
            .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
            errors="coerce",
        )
    for col in preferred_numeric:
        if col in out.columns:
            out[col] = coerce_one(out[col])

    for col in out.select_dtypes(include=["object", "string"]).columns:
        if col not in preferred_numeric:
            s = out[col].astype("string")
            if s.str.match(r"^\s*[-+]?\d+(\.\d+)?\s*$", na=False).mean() >= 0.9:
                out[col] = coerce_one(out[col])
    return out


def ensure_target_binary(
    df: pd.DataFrame, target_col: str, source_col: str = "attrition_flag"
) -> pd.DataFrame:
    """Ensure binary target column exists (0/1), deriving from source if needed."""
    out = df.copy()

    if target_col in out.columns:
        s = out[target_col]
        if pd.api.types.is_bool_dtype(s):
            out[target_col] = s.astype(int)
        elif pd.api.types.is_float_dtype(s) and set(pd.unique(s.dropna())).issubset(
            {0.0, 1.0}
        ):
            out[target_col] = s.astype(int)
        return out

    if source_col in out.columns:
        s = out[source_col].astype("string").str.lower().str.strip()
        out[target_col] = np.where(
            s.isin({"attrited customer", "attrited"}),
            1,
            np.where(s.isin({"existing customer", "existing"}), 0, np.nan),)
        out = out[out[target_col].notna()].copy()
        out[target_col] = out[target_col].astype(int)
        out.drop(columns=[source_col], inplace=True, errors="ignore")
        return out

    raise KeyError(
        f"Target '{target_col}' not found and no '{source_col}' to derive it from.")


def drop_id_duplicates_then_full(df: pd.DataFrame, id_keys: List[str]) -> pd.DataFrame:
    """Drop duplicate rows by ID first, then drop fully duplicate rows."""
    out = df.copy()
    for id_col in id_keys:
        if id_col in out.columns:
            out = out.drop_duplicates(subset=id_col, keep="first")
            break
    return out.drop_duplicates().reset_index(drop=True)
