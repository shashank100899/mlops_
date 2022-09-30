import pickle

import numpy
import pandas as pd
from pyexpat import model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("winequality-red.csv",sep=";")

x = df.drop("quality",axis=1)
y = df["quality"]

model = LinearRegression()
model = model.fit(x,y)

pickle.dump(model, open("model_file", 'wb+'))
