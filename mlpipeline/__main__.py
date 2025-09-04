from __future__ import annotations

import argparse
from pathlib import Path

from mlpipeline.pipeline import Pipeline


def main() -> None:
    """Parse CLI arguments and run the churn pipeline."""
    parser = argparse.ArgumentParser(description="Run customer churn pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/configs.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    pipeline = Pipeline(Path(args.config))
    pipeline.run()


if __name__ == "__main__":
    main()
