# ML pipeline

## Setup
### Requirements
* python 3.11
### Setup the environment and python packages
```shell
make venv PYTHON=python3.11
```
### Activate environment
```shell
source mlpipeline_env/bin/activate
```

## Project Commands
### Run the pipeline
You can run the pipeline directly using the package's `__main__.py` file:

```shell
# Run with default config
python -m mlpipeline

# Run with custom config file
python -m mlpipeline --config path/to/your/config.yaml
```

The `__main__.py` file provides a command-line interface that:
- Accepts a `--config` argument for the configuration file path
- Runs the complete pipeline using the specified configuration
- Outputs the results to the console

### Run unit test
```shell
make test
```
### Run styling
```shell
make style
```
