# mlpipeline/feature_manager.py
from __future__ import annotations

from typing import Optional

from sklearn.model_selection import train_test_split
import pandas as pd

from mlpipeline.utils.config import (
    Config, PreprocessConfig, FeatureManagerConfig,
    FeatureSpec, load_config)
from mlpipeline.utils.logger import logger


class FillMedianTransform:
    def apply(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.median())


class FillUnknownTransform:
    def apply(self, series: pd.Series) -> pd.Series:
        return series.fillna("Unknown")


class TransformFactory:
    _registry: dict[str, type] = {
        "fill_median": FillMedianTransform,
        "fill_unknown": FillUnknownTransform,
    }

    @classmethod
    def create(cls, name: str):
        transform_cls = cls._registry.get(name)
        if not transform_cls:
            raise ValueError(f"Unknown transform: {name}")
        return transform_cls()


class FeatureManager:
    """Quản lý feature cho pipeline"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._target: str = config.target_column

        # Load preprocess.yaml if provided
        self._id_keys: list[str] = []
        if config.preprocess:
            pp_cfg: PreprocessConfig = load_config(config.preprocess, PreprocessConfig)
            self._id_keys = list(pp_cfg.id_keys)

        # Load features.yaml
        self.feature_specs: list[FeatureSpec] = []
        if config.features:
            fm_cfg: FeatureManagerConfig = load_config(config.features, FeatureManagerConfig)
            self.feature_specs = fm_cfg.features

        self.feature_columns: Optional[list[str]] = None

        logger.info("FeatureManager init | target=%s | preprocess=%s",
                    self._target, self.config.preprocess)
        logger.info("id_keys=%s | features=%s", self._id_keys, self.feature_specs)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Loại ID và target, cố định list feature và fill NA.
        Trả về df chỉ gồm feature đã được impute."""
        if self._target not in df.columns:
            raise KeyError(f"Target column '{self._target}' not found in dataframe")

        # Xác định các cột feature (không gồm id_keys và target_column)
        drop_cols = set(self._id_keys + [self._target])
        keep_cols = [c for c in df.columns if c not in drop_cols]

        # Tạo bản sao DataFrame chỉ chứa feature và khoá thứ tự cột
        features_df = df[keep_cols].copy(deep=True)
        self.feature_columns = keep_cols

        # Áp dụng transform
        for spec in self.feature_specs:
            if spec.name not in features_df.columns:
                continue  # bỏ qua feature không có trong df
            transform_obj = TransformFactory.create(spec.transform)
            features_df[spec.name] = transform_obj.apply(features_df[spec.name])

        logger.info(
            "FeatureManager.transform | features=%s | X.shape=%s",
            self.feature_columns,
            features_df.shape,
        )
        return features_df

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """train/test split"""
        if not self.feature_columns:
            raise RuntimeError("Call transform() before split()")

        # Xác định X và y từ df (y = cột target)
        x = df[self.feature_columns].copy(deep=True)
        y = df[self._target].copy(deep=True)

        # Sử dụng stratify=y để giữ tỉ lệ class khi chia
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.config.valid_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        # Gộp lại thành DataFrame gồm feature và target
        train_df = x_train.copy()
        train_df[self._target] = y_train
        test_df = x_test.copy()
        test_df[self._target] = y_test
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Chạy transform -> split"""
        _ = self.transform(df)      # transform
        train_df, test_df = self.split(df)      # split
        logger.info(f"FeatureManager.train | train shape={train_df.shape}")
        logger.info(f"FeatureManager.test | test shape={test_df.shape}")
        return train_df, test_df

    def get_feature_columns(self) -> list[str]:
        """Trả danh sách cột feature đã cố định ở transform()."""
        if not self.feature_columns:
            raise RuntimeError("feature_columns is not set. Call transform() first.")
        return list(self.feature_columns)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from mlpipeline.preprocess import Preprocessor

    # Cho phép chỉ định đường dẫn config qua dòng lệnh, mặc định dùng configs/local.yaml
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/local.yaml")
    app_cfg = load_config(config_path, Config)

    print(f"[SMOKE] Đọc config: {config_path}")
    print(f"[SMOKE] Dataset: {app_cfg.dataset_path}")

    # Bước 1: Preprocess để load & clean dữ liệu
    pre = Preprocessor(app_cfg)
    clean_df = pre.run()
    print(f"[SMOKE] Dữ liệu sau clean: {clean_df.shape}")

    # Bước 2: Transform features
    fm = FeatureManager(app_cfg)
    features_df = fm.transform(clean_df)
    print(f"[SMOKE] Dữ liệu features: {features_df.shape}")
    print(f"[SMOKE] 5 cột đầu tiên: {fm.get_feature_columns()[:5]}")

    # Kiểm tra NA sau transform
    na_count = features_df.isna().sum().sum()
    print(f"[SMOKE] Số NA còn lại sau transform: {na_count}")

    # Bước 3: Chia train/test và in kích thước
    train_df, test_df = fm.run(clean_df)
    print(f"[SMOKE] Train shape: {train_df.shape}")
    print(f"[SMOKE] Test shape: {test_df.shape}")

    print("[SMOKE] Hoàn thành.")