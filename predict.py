# predict.py
import joblib
import pandas as pd
import sys

MODEL_PATH = "model/pipeline_rf_multioutput_pet.joblib"   # or your joblib model
IN = "sample_input.csv"
OUT = "predictions.csv"

def main(infile=IN, outfile=OUT):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(infile)
    # Ensure correct columns & order:
    expected = list(getattr(model, "feature_names_in_", ["Cement","Water","FineAggregate","CoarseAggregate","PET_%","w/c ratio"]))
    df = df[expected]   # will throw if missing; makes issues obvious
    preds = model.predict(df)
    # If multioutput: make nice columns
    if preds.ndim == 2 and preds.shape[1] >= 2:
        out = pd.DataFrame(preds, columns=["CS_7d","CS_28d"])
    else:
        out = pd.DataFrame(preds, columns=["prediction"])
    out.to_csv(outfile, index=False)
    print("Saved predictions to", outfile)

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv)>1 else IN
    outfile = sys.argv[2] if len(sys.argv)>2 else OUT
    main(infile, outfile)
