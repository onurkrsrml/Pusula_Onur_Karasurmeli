# Pusula Data Science Intern Case — EDA & Preprocessing

**Name:** Onur Karasürmeli  
**Email:** onurkarasurmeli.ok@gmail.com

## Overview
Prepares the Physical Medicine & Rehabilitation dataset (2235×13) for modeling with target **TedaviSuresi**.

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
Artifacts in `outputs/`:
- `dataset_cleaned.csv`
- `X_prepared.csv`
- `y_target.csv`
- `preprocess_pipeline.joblib`
- `figures/` (EDA plots)

## Method
- **Numeric:** median imputation + standardization
- **Categorical (≤20 unique):** most-frequent imputation + one-hot
- **Categorical (&gt;20 unique):** most-frequent imputation + frequency encoding
- **Multi-label:** top-20 token binarization for `KronikHastalik`, `Alerji`, `Tanilar`, `UygulamaYerleri`
- **Excluded from features:** `HastaNo` (ID), `TedaviSuresi` (target)
- **Reproducibility:** pipeline saved via `joblib`
