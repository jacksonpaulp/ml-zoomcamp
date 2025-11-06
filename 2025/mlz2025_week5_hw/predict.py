import pickle
from fastapi import FastAPI
import uvicorn

from typing import Dict, Any

app = FastAPI(title="lead-conversion-prediction")

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict_probability(datapoint: Dict[str, Any]):
    convert_probability = pipeline.predict_proba(datapoint)[0, 1]
    conversion = bool(convert_probability < .5)

    return {'conversion probability': convert_probability, 
            'conversion': conversion}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)