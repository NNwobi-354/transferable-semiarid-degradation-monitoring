# Cross-Continental Transferability of Supervised Machine Learning for Semi-Arid Land Degradation Monitoring

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NNwobi-354/transferable-semiarid-degradation-monitoring/blob/main/Transferable_Semiarid_Degradation_Monitoring_Nigeria_India.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/NNwobi-354/transferable-semiarid-degradation-monitoring)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://www.python.org/)

This repository contains the official implementation, harmonized datasets, spatial preprocessing routines, and evaluation workflows for the manuscript: **"Cross-Continental Transferability of Supervised Machine Learning for Semi-Arid Land Degradation Monitoring: From the Nigerian Sahel to the Indian Thar Desert."**

---

## 📌 Project Overview

Disentangling human-induced land degradation from climate variability in dryland ecosystems requires robust, spatially explicit predictive models. This project presents a reproducible supervised machine learning benchmark that models the **21-year RESTREND slope (2000–2020)**—a satellite-derived proxy for human-induced vegetation productivity decline—across semi-arid drylands.

Key methodological highlights include:
* **Target Variable**: 21-year residual trend analysis (RESTREND) slope isolating non-climatic vegetation productivity changes.
* **Covariate Space**: 11 harmonized eco-environmental variables spanning climate, topography, soil properties, vegetation structure, and human pressure.
* **Spatial Leakage Control**: 10 km x 10 km Spatial Block Cross-Validation preventing spatial data leakage and pseudoreplication.
* **Zero-Shot Cross-Continental Transferability**: Direct evaluation of models trained in Katsina State, Nigeria (West African Sahel) transferred to Rajasthan State, India (Thar Desert) without local retraining.
* **Model Explainability**: Game-theoretic TreeSHAP feature attributions and non-linear partial dependence analysis across biomes.

---

## 📂 Repository Structure

```text
transferable-semiarid-degradation-monitoring/
│
├── Transferable_Semiarid_Degradation_Monitoring_Nigeria_India.ipynb   # Main Google Colab Execution Notebook
├── README.md                                                          # Project documentation
│
└── data/                                                              # Root data directory
    ├── CSV/                                                           # Harmonized feature matrices
    │   ├── nigeria_katsina_dataset.csv                                # Katsina, Nigeria sampling matrix (N = 10,000)
    │   ├── india_rajasthan_dataset.csv                                # Rajasthan, India sampling matrix (N = 10,000)
    │   └── feature_metadata.csv                                       # Covariate definitions & units
    │
    ├── GEE SCRIPT/                                                    # Google Earth Engine extraction scripts
    │   ├── 01_RESTREND_Calculation_2000_2020.js                       # 21-year baseline RESTREND slope calculation
    │   ├── 02_Covariate_Extraction_Harmonization.js                   # Extraction of 11 eco-environmental variables
    │   └── 03_Spatial_Sampling_Export.js                              # Stratified grid sampling & asset export
    │
    └── SHAPEFILES/                                                    # Study area boundary files
        ├── nigeria_katsina_boundary.shp                               # Katsina State, Nigeria shapefile
        └── india_rajasthan_boundary.shp                               # Rajasthan State, India shapefile
```

---

## 📖 User Guide: How to Launch and Execute in Google Colab

This project is optimized for direct execution in **Google Colab**. All raw datasets and scripts stored in the `data/` directory (`CSV`, `GEE SCRIPT`, and `SHAPEFILES`) are automatically mounted into your session when running the notebook.

### Step 1: Open the Notebook in Google Colab
Click the **Open In Colab** badge at the top of this README or use the following direct link:
👉 [Launch Notebook in Google Colab](https://colab.research.google.com/github/NNwobi-354/transferable-semiarid-degradation-monitoring/blob/main/Transferable_Semiarid_Degradation_Monitoring_Nigeria_India.ipynb)

### Step 2: Mount Repository Directories (Cell 1)
Execute **Cell 1** in the notebook. This cell automatically:
1. Clones the GitHub repository directly into the Google Colab execution runtime.
2. Mounts all `data/` subfolders (`data/CSV/`, `data/GEE SCRIPT/`, `data/SHAPEFILES/`) into the active working directory.
3. Verifies and installs all Python dependencies (`shap`, `xgboost`, `lightgbm`, `scikit-learn`, `geopandas`).

### Step 3: Sequential Execution (Cells 2 through 6)
Once Cell 1 completes, execute Cells 2 through 6 sequentially to reproduce the research workflow:

* **Cell 1: Environment Setup & Data Mounting**
  Clones the GitHub repository, mounts the data directories, and loads the 10,000-point sampling matrices for Nigeria and India.
* **Cell 2: Exploratory Data Analysis & VIF Screening**
  Generates feature statistics, plots distributions, and computes Variance Inflation Factor (VIF) metrics to confirm low multicollinearity across the 11 covariates.
* **Cell 3: 10 km x 10 km Spatial Block Cross-Validation**
  Partitions the landscape into 10 km x 10 km spatially independent grid blocks to eliminate spatial data leakage during model training.
* **Cell 4: Supervised Model Benchmark & Hyperparameter Tuning**
  Trains and evaluates Ridge, Lasso, Random Forest, Extra Trees, XGBoost, and LightGBM models using 5-fold Spatial Block Cross-Validation.
* **Cell 5: Zero-Shot Cross-Continental Transferability**
  Applies the optimal Katsina (Nigeria) model directly to Rajasthan (India) without fine-tuning, quantifying cross-continental performance decay.
* **Cell 6: TreeSHAP Feature Attributions & Uncertainty Mapping**
  Computes game-theoretic feature importance, partial dependence curves, and spatially explicit prediction uncertainty maps.

---

## 📊 Summary of Input Datasets

All environmental covariates were processed via Google Earth Engine (GEE), spatial resolution aligned to 1 km, and sampled across 10,000 spatially stratified points per study area.

| Category | Covariate Name | Metric / Variable | Dataset Source | Native Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Target** | `RESTREND_Slope` | 21-Year RESTREND Slope (2000–2020) | MODIS MOD13A2 (NDVI) + CHIRPS v2.0 | 1 km |
| **Climate** | `Precip_Mean` | 21-Year Mean Annual Rainfall | CHIRPS v2.0 | 0.05° (~5 km) |
| **Climate** | `Precip_CV` | Rainfall Interannual Coefficient of Variation | CHIRPS v2.0 | 0.05° (~5 km) |
| **Thermal** | `LST_Day_Mean` | 21-Year Mean Daytime Land Surface Temp | MODIS MOD11A2 v061 | 1 km |
| **Thermal** | `LST_Night_Mean` | 21-Year Mean Nighttime Land Surface Temp | MODIS MOD11A2 v061 | 1 km |
| **Atmospheric**| `VPD_Mean` | 21-Year Mean Vapor Pressure Deficit | ERA5-Land Reanalysis | 0.1° (~11 km) |
| **Vegetation** | `NDVI_Baseline` | 21-Year Mean NDVI Baseline | MODIS MOD13A2 v061 | 1 km |
| **Soil** | `Soil_Organic_Carbon`| Soil Organic Carbon Stock (0–30 cm) | SoilGrids 250m v2.0 | 250 m |
| **Soil** | `Soil_Sand_Content` | Sand Fraction (0–30 cm) | SoilGrids 250m v2.0 | 250 m |
| **Topography** | `Elevation` | Surface Elevation | SRTM DEM v4 | 90 m |
| **Topography** | `Slope_Angle` | Terrain Slope | SRTM DEM v4 | 90 m |
| **Human** | `Population_Density`| Human Population Density (2020 Benchmark) | WorldPop Unconstrained | 100 m |

---

## 🔬 Key Methodological Features

### 1. Spatial Block Cross-Validation
Standard random cross-validation overestimates model performance due to spatial proximity between training and test points. We partition the study area into non-overlapping 10 km x 10 km spatial grid blocks. Whole blocks are assigned to cross-validation folds, guaranteeing strict spatial separation between training and evaluation data.

### 2. Zero-Shot Cross-Continental Evaluation
Models trained and optimized exclusively on the Katsina, Nigeria dataset are evaluated directly on the Rajasthan, India dataset without local retraining or hyperparameter adjustments. Performance decay is measured using R², RMSE, and MAE.

### 3. Explainability & Uncertainty Mapping
TreeSHAP (SHapley Additive exPlanations) is used to track feature importance shifts between biomes and map non-linear environmental thresholds. Spatially explicit prediction uncertainty is quantified through tree ensemble variance and standard deviation bounds.

---

## 🛠️ Local Installation & Environment Setup

If you prefer to run the analysis locally:

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/NNwobi-354/transferable-semiarid-degradation-monitoring.git](https://github.com/NNwobi-354/transferable-semiarid-degradation-monitoring.git)
   cd transferable-semiarid-degradation-monitoring
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install numpy pandas scikit-learn xgboost lightgbm shap geopandas matplotlib seaborn jupyter
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook Transferable_Semiarid_Degradation_Monitoring_Nigeria_India.ipynb
   ```

---

## 📜 Code & Data Availability Statement

All data extraction scripts, shapefiles, harmonized CSV matrices, and modeling workflows are fully open-access:
* **Google Earth Engine Scripts**: Located in `data/GEE SCRIPT/`
* **Study Area Shapefiles**: Located in `data/SHAPEFILES/`
* **Sampling Data Tables**: Located in `data/CSV/`

---



For questions or feedback, please open an issue on the [GitHub Issue Tracker](https://github.com/NNwobi-354/transferable-semiarid-degradation-monitoring/issues).

---
**License**: This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
