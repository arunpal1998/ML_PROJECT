import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler


df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/diabetes_prediction_dataset.csv")

gender = LabelEncoder()
smoking_history = LabelEncoder()

df['gender'] = gender.fit_transform(df['gender'])
df['smoking_history'] = smoking_history.fit_transform(df['smoking_history'])

# print(df.head())

x = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y = df['diabetes']

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=45)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = SVC(probability=True)
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print('prediction is:',pred[:4])

print('accuracy:',accuracy_score(ytest,pred))
print('confusion:',confusion_matrix(ytest,pred))
print('classification:',classification_report(ytest,pred))

gender = int(input('enter ur gender:'))
age = int(input('enter ur age:'))
hypertension = float(input('enter ur hypertension:'))
heart_disease = float(input('enter ur disease:'))
smoking_history = float(input('enter smkhistory:'))
bmi = float(input('enter ur bmi:'))
HbA1c_level = float(input('enter ur HbA1c_level:'))
blood_glucose_level = float(input('enter ur blood_glucose_level :'))

result = model.predict(scaler.transform([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]]))
print('final result is:',result)


prob = model.predict_proba(scaler.transform([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]]))
print('probabilty for diabetes positive:',prob[0][[1]])
print('probability for diabetes negative:',prob[0][0])


