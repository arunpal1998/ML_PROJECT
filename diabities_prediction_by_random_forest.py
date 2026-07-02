import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/diabetes_prediction_dataset.csv")

gender = LabelEncoder()
smoking = LabelEncoder()

df['gender'] = gender.fit_transform(df['gender'])
df['smoking_history'] = smoking.fit_transform(df['smoking_history'])

print(df.head())

x = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y = df['diabetes']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
print('prediction result Is:',pred)

print('accuracy:',accuracy_score(y_test,pred))
print('confusin:',confusion_matrix(y_test,pred))
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
print('final result is:',result[:3])

prob = model.predict_proba([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]])
print('probability for diabities positive:',prob[0][1])
print('probability fo diabities negative:',prob[0][0])