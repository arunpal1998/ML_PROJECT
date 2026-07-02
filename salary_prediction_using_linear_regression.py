import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/Salary_Data.csv")
# print(df.head())
# print(df.isna().sum())
# print(df.info())

x = df[["YearsExperience"]]
y = df["Salary"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print('final predicted result is:',pred)

print('mse:',mean_squared_error(y_test,pred))
print('mae:',mean_absolute_error(y_test,pred))
print('r2_score:',r2_score(y_test,pred))

exp = float(input('enter ur experiance:'))

result = model.predict([[exp]])
print('ur final predicted salary could be:',result)

#graph on training data
plt.scatter(x_train,y_train,color='red',label = 'actual training data')
plt.plot(x_train,model.predict(x_train), color = 'blue', label= 'regression line')
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()
plt.show()

#graph on testing data
plt.scatter(x_test,y_test,color = 'green', label='actual testing data')
plt.plot(x_test,pred, color = 'yellow',label = 'regression line')
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()
plt.show()