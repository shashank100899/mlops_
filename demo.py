import pickle

import numpy as np
import pandas as pd
from flask import Flask
from sklearn.linear_model import LinearRegression

df = pd.read_csv("winequality-red.csv",sep=";")

x = df.drop("quality",axis=1)
y = df["quality"]

model = LinearRegression()
model = model.fit(x,y)

pickle.dump(model, open("model_file", 'wb+'))

app = Flask(__name__)

@app.route("/")
def home():
    return '''<h1>mlops demo</h1> 
    <h1>Thank you !</h1>'''


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)
