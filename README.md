[Causal Transparency Framework (CTF).md](https://github.com/user-attachments/files/23882301/Causal.Transparency.Framework.CTF.md)
# Causal Transparency Framework (CTF)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Overview

The Causal Transparency Framework (CTF) is a comprehensive Python library for auditing and improving the accountability of machine learning models. It integrates causal discovery, information theory, and counterfactual analysis to move beyond correlation-based explainability to valid causal accountability.

This framework is designed for researchers, data scientists, and policymakers who need to understand not just *what* a model predicts, but *why* it makes certain decisions. CTF is particularly useful for high-stakes domains like clinical medicine, criminal justice, and finance, where fairness and transparency are critical.

## Key Features

- **Causal Discovery:** Automated discovery of causal relationships from data using a multi-algorithm consensus approach.
- **Causal Influence Index (CII):** A novel metric to quantify the causal influence of features on model predictions.
- **Complexity Measurement (CCM):** Quantify the structural complexity of models to identify the "Complexity Tax."
- **Transparency Entropy (TE):** Measure the certainty and clarity of model decisions.
- **Counterfactual Stability (CS):** Assess the robustness of model predictions to perturbations.
- **"Human Shield" Detection:** Identify when models use demographic proxies instead of actionable drivers.
- **Model Agnostic:** Works with a wide range of models, including Logistic Regression, Random Forest, XGBoost, and Deep Neural Networks.
- **Uncertainty Quantification:** Robustly estimate metrics and their sensitivity to causal graph uncertainty.

## Installation

To use the Causal Transparency Framework, you need Python 3.9+ and the following packages:

```bash
pip install numpy pandas scikit-learn xgboost torch
pip install pgmpy causallearn networkx
```

Clone the repository to get started:

```bash
git clone https://github.com/jgomarko/Causa-Transparency-framework.git
cd Causa-Transparency-framework
```

## Quick Start

Here is a quick example of how to use the CTF to analyze a model on the COMPAS dataset:

```python
from Main_experiment import CTFExperiment

# Initialize the experiment
experiment = CTFExperiment(dataset_name=\"COMPAS\", model_name=\"XGBoost\")

# Run the full analysis
results = experiment.run()

# Print the results
print(results)
```

## Documentation

### Core Modules

- **`Main_experiment.py`**: Main entry point for running CTF experiments.
- **`Ctf_metrics.py`**: Implementation of the four core CTF metrics (CII, CCM, TE, CS).
- **`causal_discovery.py`**: Causal graph discovery using a multi-algorithm consensus approach.
- **`data_preprocessing.py`**: Data loading and preprocessing for COMPAS and MIMIC-III.
- **`models.py`**: Implementations of the four model architectures (LR, RF, XGB, DNN).
- **`domain_constraints.py`**: Domain-specific causal constraints for graph discovery.
- **`enhanced_cs_perturbations.py`**: Advanced counterfactual stability perturbations.
- **`graph_validation_metrics.py`**: Metrics for validating the quality of discovered causal graphs.

### CTF Metrics

| Metric | Description | Interpretation |
| :--- | :--- | :--- |
| **CII** | Causal Influence Index | Quantifies the causal influence of features on predictions. High CII indicates a strong causal driver. |
| **CCM** | Causal Complexity Measure | Measures the structural complexity of the model. High CCM indicates a more complex, less interpretable model. |
| **TE** | Transparency Entropy | Measures the certainty of model decisions. Low TE indicates clearer, more decisive predictions. |
| **CS** | Counterfactual Stability | Assesses the robustness of predictions to perturbations. High CS indicates a more stable model. |

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite our paper:

```
[Your Paper Citation Here]
```
