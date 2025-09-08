import pandas as pd
import pytest
from mlpipeline.utils.config import Config
from mlpipeline.preprocess import Preprocessor


class TestPreprocessor:
    @pytest.fixture
    def cfg(self, tmp_path):
        # tạo file csv tạm để test load()
        csv_path = tmp_path / "dummy.csv"
        pd.DataFrame({"a": [1], "b": [2], "attrition_flag": ["Existing Customer"]}).to_csv(
            csv_path, index=False)
        return Config(
            dataset_path=str(csv_path),
            target_column="target",
            valid_size=0.2,
            random_state=42,
            model_class="mlpipeline.model.LogisticRegressionModel",
            metrics=["ACCURACY"],
        )

    def test_load_runs(self, cfg):
        pre = Preprocessor(cfg)
        df = pre.load()
        assert not df.empty
        assert list(df.columns) == ["a", "b", "attrition_flag"]

    def test_clean_runs(self, cfg):
        pre = Preprocessor(cfg)
        df = pd.DataFrame({"attrition_flag": ["Attrited Customer", "Existing Customer"]})
        out = pre.clean(df)
        assert "target" in out.columns
        assert set(out["target"].unique()) <= {0, 1}

    def test_run_pipeline_ok(self, cfg):
        pre = Preprocessor(cfg)
        out = pre.run()
        assert isinstance(out, pd.DataFrame)
        assert not out.empty
