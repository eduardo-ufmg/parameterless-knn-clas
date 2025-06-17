# Parameterless kNN Classifier

This repository contains an implementation of a parameterless k-Nearest Neighbors (kNN) classifier, along with supporting modules for convex hull computation, dissimilarity measures, kernel functions, and spatial spread analysis. The project is designed for experimentation and research in non-parametric classification methods, with a focus on eliminating the need for manual parameter tuning in kNN algorithms.

## Features
- **Parameterless kNN**: An adaptive kNN classifier that does not require manual selection of the number of neighbors.
- **Convex Hull Analysis**: Tools for geometric analysis using convex hulls.
- **Dissimilarity Measures**: Flexible dissimilarity metrics for various data types.
- **Kernel Methods**: Support for kernel-based similarity computations.
- **Spatial Spread**: Analysis of data spread in feature space.
- **Experimentation Framework**: Scripts for running experiments and storing results.
- **Preprocessed Datasets**: A collection of datasets in `.npz` format for benchmarking.

## Project Structure
```
KNN/                  # Core kNN implementation and related modules
  knn.py              # Parameterless kNN classifier
  ...
ConvexHull/           # Convex hull computation
Dissimilarity/        # Dissimilarity metrics
Kernel/               # Kernel functions
SpatialSpread/        # Spatial spread analysis
Experiment/           # Experimentation scripts and results
StoreSets/            # Dataset storage utilities
sets/                 # Preprocessed datasets (.npz files)
requirements.txt      # Python dependencies
LICENSE               # License file
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd parameterless-knn-clas
   ```
2. **Set up a virtual environment (recommended):**
   ```bash
   python3 -m venv pyvenv
   source pyvenv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- **Running Experiments:**
  - Use the scripts in the `Experiment/` directory to run classification experiments on the provided datasets.
  - Example:
    ```bash
    python Experiment/experiment.py
    ```
- **Datasets:**
  - Preprocessed datasets are available in the `sets/` directory in `.npz` format.

## Modules Overview
- `KNN/knn.py`: Main parameterless kNN classifier implementation.
- `ConvexHull/convex_hull.py`: Convex hull utilities.
- `Dissimilarity/dissimilarity.py`: Dissimilarity functions.
- `Kernel/kernel.py`: Kernel methods.
- `SpatialSpread/spatial_spread.py`: Spatial spread analysis.
- `Experiment/experiment.py`: Experiment runner.
- `StoreSets/store_sets.py`: Dataset management.

## Pre-commit Hooks
This project uses [pre-commit](https://pre-commit.com/) to ensure code quality and consistency. Pre-commit hooks automatically format and sort imports before each commit using `black` and `isort`.

### Setup Instructions
1. **Install pre-commit (if not already installed):**
   ```bash
   pip install pre-commit
   ```
2. **Install the git hooks:**
   Run this command in the root of the repository:
   ```bash
   pre-commit install
   ```
   This will set up the hooks defined in `.pre-commit-config.yaml`.

3. **Run hooks manually (optional):**
   To check all files, run:
   ```bash
   pre-commit run --all-files
   ```

### What hooks are used?
- **black**: Formats Python code to the Black style.
- **isort**: Sorts Python imports automatically.

You can configure or update hooks in `.pre-commit-config.yaml`.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Developed at UFMG.
- Includes datasets from UCI Machine Learning Repository and other sources.

## Contact
For questions or contributions, please open an issue or contact the maintainer.
