# Diversification as Risk-Minimization - Supplementary Meterial

## Main Text _with Appendix_ (PDF) is [HERE](https://raw.githubusercontent.com/anonymousIR26/wsdm-sup/main/paper.pdf)
The PDF contains the **main text** and a detailed **appendix** with **proofs, additional experiments, and limitations**.

- **Main Text (start)** — [HERE](https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2FanonymousIR26%2Fwsdm-sup%2Fmain%2Fpaper.pdf#page=1)
- **Limitations** — [HERE](https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2FanonymousIR26%2Fwsdm-sup%2Fmain%2Fpaper.pdf#page=11)
- **Proofs** — [HERE](https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2FanonymousIR26%2Fwsdm-sup%2Fmain%2Fpaper.pdf#page=11)
- **Additional Experiments** — [HERE](https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2FanonymousIR26%2Fwsdm-sup%2Fmain%2Fpaper.pdf#page=12)

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
