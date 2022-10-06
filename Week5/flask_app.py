

import pickle
from flask import Flask, request, jsonify

#Load the model and dv files

with open('model1.bin','rb') as f_in:
    model = pickle.load(f_in)


with open('dv.bin','rb') as f_in:
    dv = pickle.load(f_in)


app = Flask('card')


@app.route('/ping', methods=['GET'])
def ping():
    print('Pong')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    ypred = model.predict_proba(X)[0,1]
    card = ypred >= 0.5
    result = {
        "card probability": float(ypred),
        "card decision" : bool(card)
    }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=9696, host="0.0.0.0")