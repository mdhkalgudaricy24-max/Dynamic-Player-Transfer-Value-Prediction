"""Predict market value (EUR millions) for players in a CSV that has the same columns as
data/all_in_one_dataset.csv.   Usage: python src/predict.py data/all_in_one_dataset.csv"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train  # noqa: E402

model = joblib.load(ROOT / "models" / "best_model.joblib")
df = pd.read_csv(sys.argv[1])
names = df["player_name"] if "player_name" in df else pd.Series(range(len(df)))
X = df.drop(columns=[c for c in train.NON_FEATURES + [train.TARGET] if c in df.columns])
X["injury_prone_body_parts"] = X["injury_prone_body_parts"].fillna("None")
X = X[model.regressor_.feature_names_in_]   # keep exactly the columns the model was trained on
print(pd.DataFrame({"player": names, "predicted_value_eur_m": model.predict(X).round(2)}).head(10).to_string(index=False))
