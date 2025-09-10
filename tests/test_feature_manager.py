import pandas as pd
import pytest

from mlpipeline.utils.config import Config
from mlpipeline.feature_manager import FeatureManager


class TestFeatureManagerPR1:
    @pytest.fixture
    def cfg(self, tmp_path):
        csv_path = tmp_path / "dummy.csv"
        pd.DataFrame({
            "clientnum": [1, 2, 3, 4, 5, 6],
            "a": [10, 20, 30, 40, 50, 60],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "target": [0, 1, 0, 1, 0, 1],  # Mỗi lớp có 3 mẫu
        }).to_csv(csv_path, index=False)
        return Config(
            dataset_path=str(csv_path),
            target_column="target",
            valid_size=0.2,
            random_state=42,
            model_class="mlpipeline.model.LogisticRegressionModel",
            metrics=["ACCURACY"],
        )

    def test_transform_and_get_features(self, cfg):
        fm = FeatureManager(cfg)
        df = pd.read_csv(cfg.dataset_path)

        # TH1: có id_keys
        fm._id_keys = ["clientnum"]
        X = fm.transform(df)
        assert "target" not in X.columns
        assert "clientnum" not in X.columns
        assert fm.get_feature_columns() == list(X.columns)

        # TH2: không có id_keys
        fm2 = FeatureManager(cfg)
        fm2._id_keys = []
        X2 = fm2.transform(df)
        assert "target" not in X2.columns
        assert fm2.get_feature_columns() == list(X2.columns)

    def test_split_and_run(self, cfg):
        df = pd.read_csv(cfg.dataset_path)

        # Kiểm tra split() sau khi transform()
        fm = FeatureManager(cfg)
        fm.transform(df)
        train_df, test_df = fm.split(df)
        # Tổng số dòng phải bằng số dòng gốc
        assert len(train_df) + len(test_df) == len(df)
        # Cột target vẫn tồn tại trong train/test
        assert cfg.target_column in train_df.columns
        assert cfg.target_column in test_df.columns
        # train/test không trống (với valid_size=0.2 và 2 dòng sẽ là 1/1)
        assert not train_df.empty
        assert not test_df.empty

        # Kiểm tra run() orchestrate transform → split
        fm2 = FeatureManager(cfg)
        train2, test2 = fm2.run(df)
        assert len(train2) + len(test2) == len(df)
        assert cfg.target_column in train2.columns
        assert cfg.target_column in test2.columns
        assert fm2.feature_columns is not None
