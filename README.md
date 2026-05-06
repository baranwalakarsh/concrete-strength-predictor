# PET Concrete Strength Prediction Model

This package contains a trained machine learning model to predict
7-day and 28-day compressive strength of PET-modified concrete.

## Files
- pipeline_rf_multioutput_pet.joblib : Trained ML model
- predict.py : Script to run predictions
- sample_input.csv : Example input file
- requirements.txt : Required Python libraries

## Input Features
The CSV file must contain these columns:

Cement, Water, FineAggregate, CoarseAggregate, PET_%, w/c ratio

## Output
The output CSV will contain:
- CS_7d : 7-day compressive strength
- CS_28d : 28-day compressive strength

## How to Run (Linux / macOS)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python predict.py sample_input.csv predictions.csv
