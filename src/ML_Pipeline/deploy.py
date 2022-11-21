import pandas as pd
from flask import Flask
from flask import request
import json
import Preprocess
import Predict
import util

app = Flask(__name__)

model_path = '../output/autoencoder-model'
ml_model, columns = util.load_model(model_path)

@app.post("/get_fraud_score")
async def get_fraud_score():
    items = json.loads(request.data)
    test_df = pd.DataFrame([items], columns=items.keys())
    preprocessed_df = Preprocess.preprocess_data(test_df, is_train=False)
    prediction = Predict.init(preprocessed_df, ml_model, columns)
    output = {"Score": prediction}
    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)