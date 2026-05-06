from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model/pipeline_rf_multioutput_pet.joblib")

class InputRow(BaseModel):
    Cement: float
    Water: float
    FineAggregate: float
    CoarseAggregate: float
    PET_: float    # JSON can't have % easily - use PET_ or PET_pct
    w_c_ratio: float

@app.post("/predict")
def predict(row: InputRow):
    df = pd.DataFrame([{
        "Cement": row.Cement,
        "Water": row.Water,
        "FineAggregate": row.FineAggregate,
        "CoarseAggregate": row.CoarseAggregate,
        "PET_%": row.PET_,          # adapt if your model expects 'PET_%'
        "w/c ratio": row.w_c_ratio  # adapt key names as needed
    }])
    preds = model.predict(df)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return {"CS_7d": float(preds[0,0]), "CS_28d": float(preds[0,1])}
    return {"prediction": float(preds.flatten()[0])}
