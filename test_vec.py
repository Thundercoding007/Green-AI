import joblib
from pathlib import Path

path = Path("data/models/green/vectorizer.pkl")

print("Loading:", path)

vec = joblib.load(path)
print("Loaded OK.")

print("Type:", type(vec))

try:
    print("Num features:", len(vec.get_feature_names_out()))
except Exception as e:
    print("ERROR getting features:", e)
