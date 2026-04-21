# WATER: Wide-swath AlTimetry for Eddy Reconstruction (WP2)

This repository contains the Machine Learning implementation for **Work Package 2** of the WATER project, funded by the European Space Agency (ESA).

The goal of this package is to transition from physical-constraint-based detection to a data-driven approach using Deep Learning (U-Net) to identify "active" mesoscale eddies.

## Project Overview
The WATER project aims to quantify and enrich the mesoscale content extracted from conventional altimetry and experimental SWOT-enriched data. 

**Active Eddies** are defined as coherent sea-level anomaly patterns that possess a significant, co-localized environmental anomaly in Sea Surface Temperature (SST). This approach filters out spurious noise and focuses on physically/biologically active structures.

## Model Architecture
We implement a multi-task **U-Net** that simultaneously performs:
1. **Binary Classification**: Detection of the eddy mask.
2. **Regression**: Reconstruction of the co-localized SST spatial anomaly.

### Custom Loss Functions
To ensure physical consistency, the model is trained using a composite loss function:
- **Weighted BCE**: For the eddy mask.
- **MSE**: For the SST anomaly reconstruction.
- **Boundary-SST Coupling Loss**: A custom term that forces the mask branch to align its borders with high SST gradients.
- **Gradient Loss**: Ensures sharp edges in the reconstructed temperature fields.

## Performance Highlights
Based on current evaluations (see `eval/`):
- **Nadir-U-Net**: Achieves high F1-scores (~0.81) on conventional altimetry.
- **SWOT-U-Net**: Fine-tuned on SWOT-enriched data, detecting up to **30% more eddies** compared to dynamical algorithms when applied to nadir data.
- **Explainability**: The model leverages the SST anchor to identify high-resolution patterns even from low-resolution altimetry inputs.

## Repository Structure
- `src/`: Core library (Model, Losses, Data Generators).
- `experiments/`: Training scripts for Nadir (full/pre-train) and SWOT (fine-tuning).
- `eval/`: Scripts for binary/regression metrics and logit threshold optimization.

## Installation
```bash
git clone https://github.com/YourUsername/WATER-ML-Eddy-Detection.git
cd WATER-ML-Eddy-Detection
pip install -r requirements.txt