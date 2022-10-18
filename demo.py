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
import mlflow

now = datetime.now()
time_stamp = now.strftime("%d:%m:%Y_%H:%M:%S")

def metrics(predicted,actual):
    mae = mean_absolute_error(actual,predicted)
    mse = mean_squared_error(actual,predicted)
    r2 = r2_score(actual,predicted)

    return mae , mse , r2

df = pd.read_csv("winequality-red.csv",sep=";")

x = df.drop("quality",axis=1)
y = df[["quality"]]

standard_model = StandardScaler()
standard_model.fit(x)
x_trained = standard_model.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_trained,y) 

with mlflow.start_run():
    model = LinearRegression()
    model = model.fit(x_train,y_train)
    predicted = model.predict(x_test)

    mae , mse , r2 = metrics(y_test , predicted)

    mlflow.log_metric("mae" , mae)
    mlflow.log_metric("mse" , mse)
    mlflow.log_metric("r2"  ,r2)

    mlflow.sklearn.log_model(model , "model")

