# Customer Intelligence System

This repository contains a Python-based Customer Intelligence System implemented in `customer_intelligence.py`.

## Project Overview

The script performs end-to-end analytics using classification, ensemble learning, and clustering techniques. It reads country-level data from `Country-data.csv`, derives development-related labels, trains classification models, and produces customer (country) segments for actionable insights.

### Key components
- Data loading and exploratory analysis
- Feature engineering
- Classification using:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Hyperparameter tuning with `GridSearchCV`
- Clustering using:
  - K-Means
  - DBSCAN
- Visualization of results and model performance
- Export of segmented output and cluster profile data

## Requirements

Install the required Python packages before running the script:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage

From the project directory, run:

```powershell
python customer_intelligence.py
```

The script will generate the following outputs:

- `customer_intelligence_results.csv`
- `kmeans_cluster_profiles.csv`
- `plot_distributions.png`
- `plot_heatmap.png`
- `plot_boxplots.png`
- `plot_label_dist.png`
- `plot_model_comparison.png`
- `plot_cm_Logistic_Regression.png`
- `plot_cm_Random_Forest.png`
- `plot_cm_XGBoost.png`
- `plot_feature_importance.png`
- `plot_elbow.png`
- `plot_kmeans_pca.png`
- `plot_dbscan_eps.png`
- `plot_dbscan_pca.png`
- `plot_radar.png`

## Notes

- The current dataset is `Country-data.csv`, which contains country-level indicators.
- The model target is a derived development label (`Underdeveloped`, `Developing`, `Developed`).
- If you want to adapt this to customer intelligence, replace the dataset with customer-level data and update feature engineering accordingly.

## Project Files

- `customer_intelligence.py` — main analysis script
- `Country-data.csv` — dataset used by the script
- `README.md` — project documentation
