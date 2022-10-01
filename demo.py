import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

now = datetime.now()
time_stamp = now.strftime("%d:%m:%Y_%H:%M:%S")

def metrics(predicted,actual):
    mae = mean_absolute_error(actual,predicted)
    mse = mean_squared_error(actual,predicted)
    r2 = r2_score(actual,predicted)

    s = f"mean absoluate error = {mae} \n mean squared error = {mse} \n r square = {r2}"

    return s

df = pd.read_csv("winequality-red.csv",sep=";")

x = df.drop("quality",axis=1)
y = df[["quality"]]

standard_model = StandardScaler()
standard_model.fit(x)
x_trained = standard_model.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_trained,y) 

model = LinearRegression()
model = model.fit(x_train,y_train)
predicted = model.predict(x_test)

if os.path.isdir("models"):
    pass
else:
    os.mkdir("models")

if os.path.isdir("metrics"):
    pass
else:
    os.mkdir("metrics")


model_path = os.path.join("models", "model_file-" + str(time_stamp))
metric_path = os.path.join("metrics", "value_file-" + str(time_stamp) + ".txt")

with open(model_path, 'wb+') as f: 
    pickle.dump(model, f)

with open("metrics/vlues.txt" , 'w+') as value_file:
    value_file.write(metrics(predicted,y_test))
