import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('diabetes_prediction_model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data).reshape(1, -1)
        
        # Perform prediction using loaded model
        result = model.predict(data)
        
        # Interprete the prediction result
        interpretation = "Unknown"
        if result[0] == 1:
            interpretation = "The patient has a high chance of diabetes."
        elif result[0] == 0:
            interpretation = "The patient is non-diabetic."
        
        # We can return the result as dictionary or any desired format
        return interpretation
        
    except Exception as e:
        return json.dumps({"error":str(e)})