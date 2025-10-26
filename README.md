# Code for _Diversification as Risk-Minimization_, full paper at WSDM 2026


## How to Run the Code

To run the experiments, use the `run.py` script in the `src` directory. You can sweep through different hyperparameters by specifying a parameter and optionally a metric.

### Requirements

```bash
pip install -r requirements.txt
```

### Usage

```bash
python src/run.py <param> [metric] [-c CONFIG] [-o OUTDIR] [-d DATASET]
```

**Example:**

To run a sweep for the `beta` parameter using the metric defined in the YAML configuration file:

```bash
python src/run.py beta
```

### Configuration

The experiment parameters are defined in `src/config/config.yaml`. You can modify this file to change the default dataset, metric, and other parameters. The `experiment_params` section defines the values to be used when sweeping a hyperparameter.

### Note
- Note that MovieLens 32M data is not included in the repository due to its size. You can download it from [here](https://grouplens.org/datasets/movielens/32m/).
- Note that the MMR method is required to use the document data, which is not included due to permission issues. Thus, if you run MMR, the code will measure similarity between documents only based on the intent.
