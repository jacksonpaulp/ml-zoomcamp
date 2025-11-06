import pickle
from fastapi import FastAPI
import uvicorn

from typing import Dict, Any

app = FastAPI(title="customer-churn-prediction")

# datapoint = {
#     'gender': 'female',
#     'seniorcitizen': 0,
#     'partner': 'yes',
#     'dependents': 'no',
#     'phoneservice': 'no',
#     'multiplelines': 'no_phone_service',
#     'internetservice': 'dsl',
#     'onlinesecurity': 'no',
#     'onlinebackup': 'yes',
#     'deviceprotection': 'no',
#     'techsupport': 'no',
#     'streamingtv': 'no',
#     'streamingmovies': 'no',
#     'contract': 'month-to-month',
#     'paperlessbilling': 'yes',
#     'paymentmethod': 'electronic_check',
#     'tenure': 1,
#     'monthlycharges': 29.85,
#     'totalcharges': 29.85
# }


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict_probability(datapoint: Dict[str, Any]) :
    churn_probability = pipeline.predict_proba(datapoint)[0, 1]
    churn = bool(churn_probability < .5)

    return {'churn probability': churn_probability, 
            'churn': churn} 


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)