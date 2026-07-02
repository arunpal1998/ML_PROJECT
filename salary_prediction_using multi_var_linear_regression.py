import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/multiple_linear_regression_dataset (1).csv")

print(df.head())
# print(df.isna().sum())

x = df[["age","experience"]]
y = df["income"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

pred = model.predict(x_test)
print('predicted value is:',pred)

print('mse:',mean_squared_error(y_test,pred))
print('mae:',mean_absolute_error(y_test,pred))
print('r2_score:',r2_score(y_test,pred))

age = int(input('enter ur age:'))
exp = int(input('enter ur experience:'))
result = model.predict([[age,exp]])
print('final resutl is:',result)
