import joblib
import os 
import json
import numpy as np


def init():
    global model
    model_path = os.path.join("outputs","demo_model.pkl")
    model = joblib.load(open(model_path,"rb"))


def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    data = np.array(data)
    result = model.predict(data)
    # Log the input and output data to appinsights:
    info = {
        "input": raw_data,
        "output": result.tolist()
        }
    print(json.dumps(info))
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

    return {"result": result.tolist()}