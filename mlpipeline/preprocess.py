# mlpipeline/preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Final, List, Set
import re
import numpy as np
import pandas as pd

from mlpipeline.utils.config import Config
from mlpipeline.utils.logger import logger


class Preprocessor:
    """
    Issue 1: Load → Clean (data hygiene). 
    - Chuẩn hoá cột (snake_case)
    - Làm sạch cell: bỏ token rác (@@@, $$$, ###), chuẩn hoá NA-string → NaN
    - Gỡ '_err' trong các cột số & ép về numeric
    - Tạo 'target' 0/1 từ Attrition_Flag (nếu target chưa tồn tại)
    - Drop duplicates (ưu tiên theo client id nếu có), drop constant columns
    """

    def __init__(self, config: Config) -> None:
        self.config: Final[Config] = config

        # Các cột số thường gặp ở churn dataset — ép numeric nếu có mặt
        self._numeric_keys: List[str] = [
            "clientnum",
            "customer_age",
            "dependent_count",
            "months_on_book",
            "total_relationship_count",
            "months_inactive_12_mon",
            "contacts_count_12_mon",
            "credit_limit",
            "total_revolving_bal",
            "avg_open_to_buy",
            "total_amt_chng_q4_q1",
            "total_trans_amt",
            "total_trans_ct",
            "total_ct_chng_q4_q1",
            "avg_utilization_ratio",
        ]

        # Cột ID nếu có (dùng để drop_duplicates theo id trước)
        self._id_keys: List[str] = ["clientnum"]

        # Cột phân loại cần chuẩn hoá text nhẹ
        self._categorical_keys_hint: List[str] = [
            "gender",
            "education_level",
            "marital_status",
            "income_category",
            "card_category",
        ]

    # ===================== PUBLIC =====================
    def run(self) -> pd.DataFrame:
        """Orchestrate: LOAD → CLEAN."""
        logger.info("Preprocessor.run() started [load -> clean]")
        df_raw = self.load()
        df_clean = self.clean(df_raw)
        logger.info("Preprocessor.run() done: raw=%s -> clean=%s", df_raw.shape, df_clean.shape)
        return df_clean

    def load(self) -> pd.DataFrame:
        """Đọc CSV theo config và trả bản copy."""
        path = Path(self.config.dataset_path)
        logger.info(f"Loading dataset from: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")

        # Đọc bình thường (không ép dtype=str); phần NA-string sẽ xử lý ở clean()
        df = pd.read_csv(path)
        logger.info(f"Dataset loaded. Shape={df.shape}")
        logger.debug(f"Columns (raw): {list(df.columns)}")
        return df.copy(deep=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Các bước làm sạch an toàn, idempotent (chạy nhiều lần không phá dữ liệu)."""
        dfx = df.copy(deep=True)

        # 1) Chuẩn hoá tên cột
        dfx = self._normalize_column_names(dfx)

        # 2) Clean ô dữ liệu: bỏ token rác + chuẩn hoá NA-string
        dfx = self._strip_text_whitespace(dfx)
        dfx = self._remove_garbage_tokens(dfx)       # @@@, $$$, ###
        dfx = self._standardize_na_tokens(dfx)       # "", "NaN", "NULL", "UNKNOWN", ...


        # 3) Chuẩn hoá cột phân loại nền tảng (title-case nhẹ, map vài case phổ biến)
        dfx = self._normalize_basic_categoricals(dfx)

        # 4) Ép numeric cho các cột “trông như số”, gỡ _err nếu có
        dfx = self._coerce_numeric_columns(dfx, preferred_numeric=self._numeric_keys)

        # Ép ID về string để không bị xử lý như số ở các bước sau
        for id_col in self._id_keys:
            if id_col in dfx.columns:
                dfx[id_col] = dfx[id_col].astype("string")

        # 6) Đảm bảo target có mặt & là 0/1
        dfx = self._ensure_target_binary(dfx)

        # 7) Drop duplicates (ưu tiên theo client id nếu có)
        dfx = self._drop_id_duplicates_then_full(dfx, id_keys=self._id_keys)

        logger.debug(f"Columns (clean): {list(dfx.columns)}")
        return dfx

    # ===================== HELPERS =====================
    @staticmethod
    def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        def norm(c: str) -> str:
            nc = c.strip().lower()
            nc = re.sub(r"[^\w]+", "_", nc)
            nc = re.sub(r"_+", "_", nc).strip("_")
            return nc
        out = df.copy()
        out.columns = [norm(c) for c in out.columns]
        return out

    @staticmethod
    def _strip_text_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.select_dtypes(include=["object", "string", "category"]).columns:
            out[c] = out[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        return out

    @staticmethod
    def _remove_garbage_tokens(df: pd.DataFrame) -> pd.DataFrame:
        """Bỏ token rác như @@@, $$$, ###."""
        out = df.copy()
        pattern = re.compile(r"(@@@|\$\$\$|###)")
        for c in out.columns:
            if pd.api.types.is_string_dtype(out[c]):
                out[c] = out[c].astype("string").str.replace(pattern, "", regex=True).str.strip()
        return out

    @staticmethod
    def _standardize_na_tokens(df: pd.DataFrame) -> pd.DataFrame:
        """Đưa các NA-string phổ biến về np.nan (''/NaN/NULL/UNKNOWN/NONE...)."""
        out = df.copy()
        na_tokens: Set[str] = {"", "na", "n/a", "null", "none", "nan", "unknown"}
        for c in out.columns:
            if pd.api.types.is_string_dtype(out[c]):
                s = out[c].astype("string")
                out[c] = s.where(~s.str.lower().isin(na_tokens), other=np.nan)
        return out

    def _normalize_basic_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hoá nhẹ cho một số cột phân loại: title-case, map vài case phổ biến."""
        out = df.copy()

        def tidy_series(s: pd.Series) -> pd.Series:
            s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            return s

        # Education_Level
        col = "education_level"
        if col in out.columns:
            s = tidy_series(out[col])
            edu_map = {
                "uneducated": "Uneducated",
                "high school": "High School",
                "college": "College",
                "graduate": "Graduate",
                "post graduate": "Post-Graduate",
                "post-graduate": "Post-Graduate",
                "doctorate": "Doctorate",
            }
            out[col] = s.str.lower().map(edu_map).fillna(s.str.title())

        # Marital_Status
        col = "marital_status"
        if col in out.columns:
            s = tidy_series(out[col])
            mar_map = {"married": "Married", "single": "Single", "divorced": "Divorced"}
            out[col] = s.str.lower().map(mar_map).fillna(s.str.title())

        # Income_Category: fix một vài biến thể
        col = "income_category"
        if col in out.columns:
            s = tidy_series(out[col])
            out[col] = s.str.replace("Less Than $40K", "Less than $40K", regex=False)

        return out

    @staticmethod
    def _coerce_numeric_columns(df: pd.DataFrame, preferred_numeric: List[str]) -> pd.DataFrame:
        """Gỡ `_err` và ép numeric cho các cột trong list; nếu không có trong list vẫn thử heuristic."""
        out = df.copy()

        def coerce_one(series: pd.Series) -> pd.Series:
            s = series.astype("string")
            s = s.str.replace(r"_err\b", "", regex=True)
            # lấy số đầu tiên (giữ dấu & thập phân)
            s = s.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
            return pd.to_numeric(s, errors="coerce")

        # Ưu tiên cột trong preferred list
        for col in preferred_numeric:
            if col in out.columns:
                out[col] = coerce_one(out[col])

        # Heuristic cho cột object khác có >90% pattern số
        for col in out.select_dtypes(include=["object", "string"]).columns:
            if col in preferred_numeric:
                continue
            s = out[col].astype("string")
            valid = s.str.match(r"^\s*[-+]?\d+(\.\d+)?\s*$", na=False)
            if valid.mean() >= 0.9:
                out[col] = coerce_one(out[col])

        return out

    def _ensure_target_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Đảm bảo cột target (config.target_column) tồn tại và là 0/1.
        - Nếu đã có: giữ nguyên (nhưng chuẩn hoá 0/1 nếu là bool/float 0.0/1.0)
        - Nếu chưa: tìm 'attrition_flag' để map -> target
        """
        out = df.copy()
        tgt = self.config.target_column

        if tgt in out.columns:
            s = out[tgt]
            # chuẩn 0/1 nếu là bool/float 0.0/1.0
            if pd.api.types.is_bool_dtype(s):
                out[tgt] = s.astype(int)
            elif pd.api.types.is_float_dtype(s):
                uniq = set(pd.unique(s.dropna()))
                if uniq.issubset({0.0, 1.0}):
                    out[tgt] = s.astype(int)
            return out

        # Chưa có target → thử map từ attrition_flag
        candidates = [c for c in out.columns if c.lower() == "attrition_flag"]
        if candidates:
            col = candidates[0]
            s = out[col].astype("string").str.lower().str.strip()
            out[tgt] = np.where(s.isin({"attrited customer", "attrited"}), 1,
                         np.where(s.isin({"existing customer", "existing"}), 0, np.nan))
            n_missing = int(out[tgt].isna().sum())
            if n_missing > 0:
                logger.warning(f"Target '{tgt}' has {n_missing} NaN after mapping from '{col}'. Dropping those rows.")
                out = out[out[tgt].notna()].copy()
            out[tgt] = out[tgt].astype(int)
            if col in out.columns:
                out = out.drop(columns=[col])
            return out

        raise KeyError(
            f"Target column '{tgt}' not found and no 'attrition_flag' to derive it from."
        )

    @staticmethod
    def _drop_id_duplicates_then_full(df: pd.DataFrame, id_keys: List[str]) -> pd.DataFrame:
        out = df.copy()
        for id_col in id_keys:
            if id_col in out.columns:
                before = len(out)
                out = out.drop_duplicates(subset=id_col, keep="first")
                dropped = before - len(out)
                if dropped > 0:
                    logger.info(f"Dropped {dropped} duplicate rows by id '{id_col}'.")
                break  # chỉ dùng key đầu tiên tồn tại
        # Tiếp tục drop duplicates toàn hàng (nếu còn)
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        dropped = before - len(out)
        if dropped > 0:
            logger.info(f"Dropped {dropped} fully-duplicate rows.")
        return out


# ===== CLI: chạy từng bước như notebook =====
if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/configs.yaml",
                        help="Path to YAML config (default: configs/configs.yaml)")
    parser.add_argument("--step", choices=["load", "clean", "run"], default="run",
                        help="Run a single step or the whole pipeline")
    parser.add_argument("--preview", action="store_true",
                        help="Print a quick preview of the resulting dataframe")
    parser.add_argument("--rows", type=int, default=5,
                        help="Rows to show in preview head()")
    args = parser.parse_args()

    cfg = Config(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    pre = Preprocessor(cfg)

    if args.step == "load":
        out = pre.load()
    elif args.step == "clean":
        out = pre.clean(pre.load())
    else:  # run
        out = pre.run()

    if args.preview:
        print("Shape:", out.shape)
        print("Columns:", list(out.columns))
        pd.set_option("display.max_columns", None)  # hiện tất cả cột
        pd.set_option("display.width", 0)           # fit ra nhiều dòng terminal
        pd.set_option("display.max_rows", 20)       # nếu muốn xem nhiều dòng
        print(out.head(args.rows))
        print("\nColumn dtypes:")
        print(out.dtypes)

