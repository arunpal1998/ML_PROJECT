import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/diabetes_prediction_dataset.csv")

# print(df.isna().sum())

# df = df.drop_duplicates()
# print(df.duplicated().sum())

gender = LabelEncoder()
smoke = LabelEncoder()

df['gender'] = gender.fit_transform(df['gender'])
df['smoking_history'] = smoke.fit_transform(df['smoking_history'])
print(df.head())
print(df.isna().sum())

x = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y = df['diabetes']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print('predicted result is:',pred[:5])

print('accuracy:',accuracy_score(y_test,pred))
print('confusion:',confusion_matrix(y_test,pred))
print('classification report:',classification_report(y_test,pred))

gender = int(input('enter ur gender:'))
age = int(input('enter ur age:'))
hypertension = float(input('enter ur hypertension:'))
heart_disease = float(input('enter ur disease:'))
smoking_history = float(input('enter smkhistory:'))
bmi = float(input('enter ur bmi:'))
HbA1c_level = float(input('enter ur HbA1c_level:'))
blood_glucose_level = float(input('enter ur blood_glucose_level :'))

result = model.predict([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]])

print('final result is:',result[:5])

prob = model.predict_proba([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]])

print('probability for diabities negative is:',prob[0][0])
print('probability for diabities positive is:',prob[0][1])
