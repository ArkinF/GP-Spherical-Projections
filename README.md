# Gaussian Process Regression with Spherical Projections

This repository contains the implementation and experiments for my MSc thesis on Gaussian Process regression using spherical projection methods.

## Overview

The project implements and compares three Gaussian Process approaches:
- **Exact GP**: Standard Gaussian Process regression
- **Spherical Projection GP**: Novel approach using spherical projections for scalability
- **SVGP**: Sparse Variational Gaussian Process

## Repository Structure

```
├── src/                    # Source code
│   ├── models/            # GP model implementations
│   ├── kernels/           # Kernel functions
│   ├── training/          # Training utilities
│   └── utils/             # Helper functions
├── notebooks/             # Jupyter notebooks for experiments
├── data/                  # Dataset files
├── results/               # Generated figures and results
└── requirements.txt       # Python dependencies
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run experiments via main script:
```bash
python main.py --experiment toy_1d
python main.py --experiment toy_2d
python main.py --experiment d_sweep
```

3. Or use Jupyter notebooks:
```bash
cd notebooks
jupyter lab
```

## Key Features

- **Modular Design**: Clean separation of models, kernels, and training code
- **Comprehensive Testing**: Multiple datasets and experimental setups
- **Visualisation**: Built-in plotting utilities for posterior analysis
- **Scalability**: Efficient implementations for large-scale datasets

## Datasets

- **Toy Examples**: 1D and 2D synthetic data
- **Mauna Loa CO2**: Atmospheric CO2 concentration data
- **Daily Min Temperatures**: Melbourne temperature data
- **CASP Protein**: Protein tertiary structure prediction
- **Power Plant**: Combined cycle power plant data
- **Sunspots**: Monthly sunspot activity data
- **CCPP**: Combined cycle power plant data

## Results

The experiments demonstrate the effectiveness of spherical projection methods in maintaining accuracy while improving computational efficiency compared to exact GP methods.
