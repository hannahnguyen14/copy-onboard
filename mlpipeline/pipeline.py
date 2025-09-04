from __future__ import annotations

from pathlib import Path

# from mlpipeline.evaluate import Evaluator
# from mlpipeline.feature_manager import FeatureManager
# from mlpipeline.model import MLModel
# from mlpipeline.preprocess import Preprocessor
from mlpipeline.utils.config import load_config
# from mlpipeline.utils.factory import create
from mlpipeline.utils.logger import logger


class Pipeline:
    """Orchestrates the complete machine learning pipeline workflow."""

    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)
        # self.preprocessor = self._get_preprocessor()
        # self.model = self._get_model()
        # self.evaluator = self._get_evaluator()
        # self.feature_manager = self._get_feature_manager()

    # def _get_preprocessor(self) -> Preprocessor:
    #     return Preprocessor(self.config)

    # def _get_model(self) -> MLModel:
    #     return create(self.config.model_class)()

    # def _get_evaluator(self) -> Evaluator:
    #     return Evaluator(self.config.metrics)

    # def _get_feature_manager(self) -> FeatureManager:
    #     return FeatureManager(self.config)

    def run(self) -> None:
        """Run the pipeline."""
        logger.info("Pipeline: Running pipeline")
        # df = self.preprocessor.run()

        # logger.info("Pipeline: Saving features to feature manager")
        # train_df, test_df = self.feature_manager.run(df)

        # logger.info("Pipeline: Running model")
        # self.model.fit(train_df, self.feature_manager)

        # logger.info("Pipeline: Running evaluator")
        # self.evaluator.run(
        #     self.model,
        #     test_df,
        #     self.feature_manager,
        # )

        # logger.info("Pipeline: Pipeline run successfully")
